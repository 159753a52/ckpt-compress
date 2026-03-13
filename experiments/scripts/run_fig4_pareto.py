"""
Fig 4: Pareto 曲线 — 剪枝比例 vs 质量

扫描多个稀疏率，绘制每种方法的 Pareto 曲线。
展示高剪枝比例下 Ours 的优势更明显。

对比方法:
  - magnitude+uniform
  - first-order+uniform (Inshrinkerator-style)
  - second-order-hvp+uniform (ablation)
  - second-order-hvp+gamma-adaptive (Ours)

运行示例:
    python experiments/scripts/run_fig4_pareto.py \
        --model gpt2-small --dataset wikitext103 \
        --min_ratio 0.1 --max_ratio 0.9 --step 0.1 \
        --num_steps 100 --eval_batches 20 \
        --device cuda
"""

import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

import sys
from pathlib import Path
import argparse
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lib.models import load_model
from experiments.lib.data import get_data_loaders, cache_batches
from experiments.lib.evaluation import evaluate
from experiments.lib.results import save_results
from experiments.lib.importance_compare.scoring import compute_scores_by_method
from src.ckpt_compress.pruning import apply_pruning
from src.ckpt_compress.pruning.allocation import get_allocation_strategy


METHODS = [
    {'importance': 'magnitude',    'allocation': 'uniform',        'color': 'blue',  'marker': 's'},
    {'importance': 'first-order',  'allocation': 'uniform',        'color': 'green', 'marker': '^'},
    {'importance': 'second-order-hvp', 'allocation': 'uniform',        'color': 'orange','marker': 'D'},
    {'importance': 'second-order-hvp', 'allocation': 'gamma-adaptive', 'color': 'red',   'marker': 'o'},
]


def plot_pareto(results_by_method, task_type, args):
    """绘制 Pareto 曲线。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    if task_type == 'lm':
        y_key, y_label = 'perplexity', 'Perplexity'
    elif task_type == 'reg':
        y_key, y_label = 'pearson', 'Pearson'
    else:
        y_key, y_label = 'accuracy', 'Accuracy'
    
    for method_name, (results, style) in results_by_method.items():
        ratios = [r['target_ratio'] for r in results]
        values = [r[y_key] for r in results]
        ax.plot(ratios, values, label=method_name, 
                color=style['color'], marker=style['marker'], 
                linewidth=2, markersize=6)
    
    ax.set_xlabel('Pruning Ratio', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{args.model} on {args.dataset}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 如果是 perplexity，用 log scale 可能更好看
    if task_type == 'lm':
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f'fig4_{args.model}_{args.dataset}.png', dpi=300)
    plt.savefig(out_dir / f'fig4_{args.model}_{args.dataset}.pdf')
    plt.close()
    print(f"图片已保存: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Fig 4: Pareto 曲线')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--min_ratio', type=float, default=0.1)
    parser.add_argument('--max_ratio', type=float, default=0.9)
    parser.add_argument('--step', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--hvp_batches', type=int, default=8)
    parser.add_argument('--hvp_mode', type=str, default='full', choices=['full', 'block'])
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--eval_batches', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results/paper_results/fig4')
    args = parser.parse_args()
    
    prune_ratios = list(np.arange(args.min_ratio, args.max_ratio + args.step / 2, args.step))
    prune_ratios = [round(r, 2) for r in prune_ratios]
    
    print("=" * 70)
    print("Fig 4: Pareto 曲线")
    print(f"模型: {args.model} | 数据集: {args.dataset}")
    print(f"剪枝率范围: {prune_ratios}")
    print("=" * 70)
    
    # 加载
    model, _ = load_model(
        args.model,
        pretrained=True,
        checkpoint_path=args.checkpoint,
        device='cpu',
        dataset_name=args.dataset,
    )
    train_loader, val_loader, task_type = get_data_loaders(
        args.model, args.dataset, args.batch_size, args.seq_length)
    
    cached_train = cache_batches(train_loader, args.num_steps, task_type)
    cached_eval = cache_batches(val_loader, args.eval_batches, task_type)
    
    # Baseline
    model_eval = copy.deepcopy(model).to(args.device)
    baseline = evaluate(model_eval, cached_eval, task_type, args.device)
    print(f"Baseline: {baseline}")
    del model_eval
    
    # 预计算得分
    importance_methods = sorted({m['importance'] for m in METHODS})
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
    )
    del model_for_scoring
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 扫描
    all_results = []
    results_by_method = {}
    
    for m in METHODS:
        imp, alloc = m['importance'], m['allocation']
        method_name = f"{imp}+{alloc}"
        print(f"\n--- {method_name} ---")
        
        allocator = get_allocation_strategy(alloc)
        scores = score_cache[imp]
        method_results = []
        
        for ratio in prune_ratios:
            model_copy = copy.deepcopy(model).to(args.device)
            layer_ratios = allocator.allocate(scores, ratio)
            _, _, actual = apply_pruning(model_copy, scores, layer_ratios, device=args.device)
            metrics = evaluate(model_copy, cached_eval, task_type, args.device)
            
            result = {
                'method': method_name, 'importance': imp, 'allocation': alloc,
                'target_ratio': ratio, 'actual_ratio': actual,
            }
            result.update(metrics)
            result.update({f'baseline_{k}': v for k, v in baseline.items()})
            
            method_results.append(result)
            all_results.append(result)
            
            metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  ratio={ratio:.0%} | {metric_str}")
            
            del model_copy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        results_by_method[method_name] = (method_results, m)
    
    # 保存
    save_results(all_results, args.output_dir, f"fig4_{args.model}_{args.dataset}", vars(args))
    plot_pareto(results_by_method, task_type, args)
    print("\n实验完成！")


if __name__ == '__main__':
    main()
