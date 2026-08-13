"""
Table 3: 消融实验 — 拆解两个贡献的独立增益

6 种组合:
  A: magnitude + uniform             (最弱 baseline)
  B: first-order + uniform           (Inshrinkerator-style)
  C: second-order-hvp + uniform      (只加二阶 (HVP)，不加分布)
  D: first-order + gamma-adaptive    (只加分布，不加二阶)
  E: second-order-hvp + gamma-adaptive (完整方法)
  F: second-order-hvp + global-topk  (精确全局排序上限，用于验证 Gamma 近似质量)

运行示例:
    python experiments/scripts/run_ablation_study.py \
        --model gpt2-small --dataset wikitext103 \
        --prune_ratios 0.3,0.5,0.7,0.9 \
        --num_steps 100 --eval_batches 20 \
        --device cuda
"""

import os

os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

import argparse
import copy
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dacp.pruning import apply_pruning
from dacp.pruning.allocation import get_allocation_strategy
from experiments.lib.data import cache_batches, get_data_loaders
from experiments.lib.evaluation import evaluate
from experiments.lib.importance_compare.scoring import compute_scores_by_method
from experiments.lib.models import load_model
from experiments.lib.results import print_results_table, save_results

ABLATION_METHODS = [
    {"label": "A", "importance": "magnitude", "allocation": "uniform"},
    {"label": "B", "importance": "first-order", "allocation": "uniform"},
    {"label": "C", "importance": "second-order-hvp", "allocation": "uniform"},
    {"label": "D", "importance": "first-order", "allocation": "weibull-adaptive"},
    {"label": "E", "importance": "second-order-hvp", "allocation": "weibull-adaptive"},
    {"label": "F", "importance": "second-order-hvp", "allocation": "global-topk"},
]


def _loss_increase_pct(baseline_loss: float, current_loss: float) -> float:
    """Return a finite relative loss increase for a non-zero baseline."""
    baseline = float(baseline_loss)
    current = float(current_loss)
    if not math.isfinite(baseline):
        raise ValueError("baseline loss must be finite")
    if baseline == 0.0:
        raise ValueError("loss increase percentage is undefined for a zero baseline loss")
    if not math.isfinite(current):
        raise ValueError("current loss must be finite")
    return (current - baseline) / baseline * 100.0


def main():
    parser = argparse.ArgumentParser(description="Table 3: 消融实验")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prune_ratios", type=str, default="0.3,0.5,0.7,0.9")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--hvp_batches", type=int, default=8)
    parser.add_argument("--hvp_mode", type=str, default="full", choices=["full", "block"])
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="对一阶/二阶项做 per-tensor 尺度归一化",
    )
    parser.add_argument("--output_dir", type=str, default="results/paper_results/table3")
    args = parser.parse_args()

    prune_ratios = [float(r) for r in args.prune_ratios.split(",")]

    print("=" * 70)
    print("Table 3: 消融实验")
    print(f"模型: {args.model} | 数据集: {args.dataset}")
    print(f"剪枝率: {prune_ratios}")
    print("=" * 70)

    # 加载
    model, model_family = load_model(
        args.model,
        pretrained=True,
        checkpoint_path=args.checkpoint,
        device="cpu",
        dataset_name=args.dataset,
    )
    train_loader, val_loader, task_type = get_data_loaders(
        args.model, args.dataset, args.batch_size, args.seq_length
    )

    cached_train = cache_batches(train_loader, args.num_steps, task_type)
    cached_eval = cache_batches(val_loader, args.eval_batches, task_type)

    # Baseline
    model_eval = copy.deepcopy(model).to(args.device)
    baseline = evaluate(model_eval, cached_eval, task_type, args.device)
    print(f"Baseline: {baseline}")
    del model_eval

    # 预计算得分
    importance_methods = sorted({m["importance"] for m in ABLATION_METHODS})
    model_for_scoring = copy.deepcopy(model).to(args.device)
    score_cache = compute_scores_by_method(
        model_for_scoring,
        cached_train,
        task_type,
        methods=importance_methods,
        alpha=args.alpha,
        hvp_batches=args.hvp_batches,
        hvp_mode=args.hvp_mode,
        chunk_size=args.chunk_size,
        model_family=model_family,
        normalize=args.normalize,
    )
    del model_for_scoring
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 运行
    all_results = []
    for m in ABLATION_METHODS:
        label = m["label"]
        imp, alloc = m["importance"], m["allocation"]
        method_name = f"{label}: {imp}+{alloc}"
        print(f"\n--- {method_name} ---")

        allocator = get_allocation_strategy(alloc)
        scores = score_cache[imp]

        for ratio in prune_ratios:
            model_copy = copy.deepcopy(model).to(args.device)
            layer_ratios = allocator.allocate(scores, ratio)
            _, _, actual = apply_pruning(model_copy, scores, layer_ratios, device=args.device)
            metrics = evaluate(model_copy, cached_eval, task_type, args.device)

            result = {
                "label": label,
                "importance": imp,
                "allocation": alloc,
                "method": method_name,
                "target_ratio": ratio,
                "actual_ratio": actual,
            }
            result.update(metrics)
            if "loss" in baseline:
                result["loss_increase_pct"] = _loss_increase_pct(
                    baseline["loss"], metrics["loss"]
                )
            if "accuracy" in baseline:
                result["accuracy_drop"] = baseline["accuracy"] - metrics["accuracy"]

            all_results.append(result)
            print(
                f"  ratio={ratio:.0%} | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )

            del model_copy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 保存
    save_results(all_results, args.output_dir, f"table3_{args.model}_{args.dataset}", vars(args))
    print("\n结果汇总:")
    metric_key = (
        "perplexity" if task_type == "lm" else ("pearson" if task_type == "reg" else "accuracy")
    )
    print_results_table(
        all_results, ["label", "importance", "allocation", "target_ratio", "loss", metric_key]
    )
    print("\n实验完成！")


if __name__ == "__main__":
    main()
