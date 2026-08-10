"""
9.3 敏感性分析

证明方法对超参数不敏感，增强鲁棒性论证。

实验维度:
  1. 不同 compression-restore 次数 K (5, 10, 20, 50)
  2. 不同 HVP batch size (1, 4, 8, 16)
  3. 不同 bisection 收敛精度 (1e-3, 1e-6, 1e-9)

运行示例:
    python experiments/scripts/run_sensitivity_analysis.py \
        --model gpt2-medium --dataset wikitext103 \
        --prune_ratio 0.3 --device cuda
"""

import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

import sys
from pathlib import Path
import argparse
import copy
import time
import json
import torch
import numpy as np
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lib.models import load_model
from experiments.lib.data import get_data_loaders, cache_batches
from experiments.lib.evaluation import evaluate
from experiments.lib.results import save_results, print_results_table
from experiments.lib.gamma_sensitivity import (
    fit_gamma_mom_problem,
    solve_gamma_mom_rates,
)
from experiments.lib.importance_compare.scoring import compute_scores_by_method
from experiments.lib.losses import compute_task_loss
from dacp.pruning import Pruner, apply_pruning, filter_prunable_params
from dacp.pruning.allocation import (
    UniformAllocation,
    GammaAdaptiveAllocation,
)


# ============================================================
#  Helper: single train step (from run_fig6)
# ============================================================

def train_one_step(model, optimizer, batch, task_type, device):
    model.train()
    loss = compute_task_loss(model, batch, task_type, device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ============================================================
#  Experiment 1: Sweep K (compression-restore count)
# ============================================================

def sweep_K(args, model_init, train_loader, val_loader, task_type, cached_eval):
    """不同 K 下 Magnitude+Uniform vs Ours 的质量对比。"""
    K_values = [int(k) for k in args.sweep_k.split(',')]
    total_steps = args.total_steps
    results = []

    for K in K_values:
        compress_interval = total_steps // K
        for method_label in ['magnitude+uniform', 'first-order+gamma-adaptive']:
            model = copy.deepcopy(model_init).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            train_iter = iter(train_loader)

            imp_method, alloc_method = method_label.split('+')

            for step in range(total_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                train_one_step(model, optimizer, batch, task_type, args.device)

                # compress at interval
                if (step + 1) % compress_interval == 0:
                    # 收集当前权重和梯度用于评分
                    weights = {}
                    gradients = {}
                    for name, p in model.named_parameters():
                        weights[name] = p.data.detach().cpu()
                        if p.grad is not None:
                            gradients[name] = p.grad.detach().cpu()

                    from dacp.pruning.pruner import filter_prunable_params
                    prunable_w = filter_prunable_params(weights)
                    prunable_g = {k: gradients.get(k, torch.zeros_like(v)) for k, v in prunable_w.items()}

                    pruner = Pruner(importance=imp_method, allocation=alloc_method)
                    scores = pruner.compute_scores(prunable_w, prunable_g)
                    layer_ratios = pruner.compute_layer_ratios(scores, args.prune_ratio)
                    apply_pruning(model, scores, layer_ratios, device=args.device)

            metrics = evaluate(model, cached_eval, task_type, args.device)
            row = {'K': K, 'method': method_label, **metrics}
            results.append(row)
            print(f"  K={K:3d} | {method_label:35s} | "
                  + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


# ============================================================
#  Experiment 2: Sweep HVP batch size
# ============================================================

def sweep_hvp_batch(args, model_init, cached_train, cached_eval, task_type,
                    model_family='gpt2'):
    """不同 HVP batch size 对 score 质量和计算时间的影响。"""
    hvp_batches_list = [int(b) for b in args.sweep_hvp_batches.split(',')]
    results = []

    for hvp_b in hvp_batches_list:
        model = copy.deepcopy(model_init).to(args.device)
        t0 = time.perf_counter()
        score_cache = compute_scores_by_method(
            model, cached_train, task_type,
            methods=['second-order-hvp'], alpha=args.alpha,
            hvp_batches=hvp_b, hvp_mode='full', chunk_size=args.chunk_size,
            model_family=model_family,
        )
        hvp_time = time.perf_counter() - t0
        scores = score_cache['second-order-hvp']

        # 用 gamma-adaptive 分配并评估
        allocator = GammaAdaptiveAllocation()
        layer_ratios = allocator.allocate(scores, args.prune_ratio)
        model_prune = copy.deepcopy(model_init).to(args.device)
        _, _, actual = apply_pruning(model_prune, scores, layer_ratios, device=args.device)
        metrics = evaluate(model_prune, cached_eval, task_type, args.device)

        row = {'hvp_batches': hvp_b, 'hvp_time_s': round(hvp_time, 3),
               'actual_ratio': actual, **metrics}
        results.append(row)
        print(f"  hvp_batches={hvp_b:3d} | time={hvp_time:.2f}s | "
              + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        del model, model_prune
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


# ============================================================
#  Experiment 3: Sweep bisection tolerance
# ============================================================

def sweep_bisection_tol(args, model_init, cached_train, cached_eval, task_type,
                        model_family='gpt2'):
    """不同 bisection 容差对结果和速度的影响。"""
    tols = [float(t) for t in args.sweep_tol.split(',')]
    results = []

    # 先计算 scores（只算一次）
    model = copy.deepcopy(model_init).to(args.device)
    score_cache = compute_scores_by_method(
        model, cached_train, task_type,
        methods=['second-order-hvp'], alpha=args.alpha,
        hvp_batches=args.hvp_batches, hvp_mode='full',
        chunk_size=args.chunk_size,
        model_family=model_family,
    )
    scores = score_cache['second-order-hvp']
    del model

    problem = fit_gamma_mom_problem(scores)

    for tol in tols:
        t0 = time.perf_counter()
        layer_ratios, _ = solve_gamma_mom_rates(
            problem,
            args.prune_ratio,
            xtol=tol,
        )
        alloc_time = time.perf_counter() - t0

        model_prune = copy.deepcopy(model_init).to(args.device)
        _, _, actual = apply_pruning(model_prune, scores, layer_ratios, device=args.device)
        metrics = evaluate(model_prune, cached_eval, task_type, args.device)

        row = {'bisection_tol': tol, 'alloc_time_s': round(alloc_time, 5),
               'actual_ratio': actual, **metrics}
        results.append(row)
        print(f"  tol={tol:.0e} | alloc_time={alloc_time:.4f}s | "
              + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        del model_prune
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


# ============================================================
#  Experiment 4: Timing overhead breakdown
# ============================================================

def timing_breakdown(args, model_init, cached_train, cached_eval, task_type,
                     model_family='gpt2'):
    """逐阶段计时：gradient/HVP → Gamma MoM fitting → bisection → mask application。"""
    model = copy.deepcopy(model_init).to(args.device)

    # Phase 1: Gradient collection + score computation (HVP)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_score_start = time.perf_counter()
    score_cache = compute_scores_by_method(
        model, cached_train, task_type,
        methods=['second-order-hvp'], alpha=args.alpha,
        hvp_batches=args.hvp_batches, hvp_mode='full',
        chunk_size=args.chunk_size,
        model_family=model_family,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_score = time.perf_counter() - t_score_start
    scores = score_cache['second-order-hvp']
    del model

    # Phase 2: Gamma MoM fitting (per-layer)
    t_fit_start = time.perf_counter()
    problem = fit_gamma_mom_problem(scores)
    t_fit = time.perf_counter() - t_fit_start

    # Phase 3: Bisection for c*
    t_bisect_start = time.perf_counter()
    layer_ratios, _ = solve_gamma_mom_rates(
        problem,
        args.prune_ratio,
        xtol=1e-10,
    )
    t_bisect = time.perf_counter() - t_bisect_start
    layer_ratios = solution.layer_ratios

    # Phase 4: Mask application
    model_prune = copy.deepcopy(model_init).to(args.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_mask_start = time.perf_counter()
    _, _, actual = apply_pruning(model_prune, scores, layer_ratios, device=args.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_mask = time.perf_counter() - t_mask_start

    total = t_score + t_fit + t_bisect + t_mask

    breakdown = {
        'score_computation_s': round(t_score, 4),
        'gamma_fitting_s': round(t_fit, 6),
        'bisection_s': round(t_bisect, 6),
        'mask_application_s': round(t_mask, 4),
        'total_s': round(total, 4),
        'score_pct': round(100 * t_score / total, 1),
        'gamma_pct': round(100 * t_fit / total, 1),
        'bisect_pct': round(100 * t_bisect / total, 1),
        'mask_pct': round(100 * t_mask / total, 1),
        'num_layers': len(problem.layers),
        'actual_ratio': actual,
    }
    print(f"  Score (grad+HVP): {t_score:.3f}s ({breakdown['score_pct']}%)")
    print(f"  Gamma MoM fit:    {t_fit:.5f}s ({breakdown['gamma_pct']}%)")
    print(f"  Bisection:        {t_bisect:.5f}s ({breakdown['bisect_pct']}%)")
    print(f"  Mask application: {t_mask:.3f}s ({breakdown['mask_pct']}%)")
    print(f"  Total:            {total:.3f}s")

    del model_prune
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return breakdown


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='敏感性分析')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--prune_ratio', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--total_steps', type=int, default=500,
                        help='总训练步数（sweep K 时使用）')
    parser.add_argument('--lr', type=float, default=5e-5)

    # Sweep 参数
    parser.add_argument('--sweep_k', type=str, default='5,10,20,50',
                        help='compression-restore 次数列表')
    parser.add_argument('--sweep_hvp_batches', type=str, default='1,4,8,16',
                        help='HVP batch size 列表')
    parser.add_argument('--sweep_tol', type=str, default='1e-3,1e-6,1e-9',
                        help='bisection 容差列表')

    # 基础参数
    parser.add_argument('--hvp_batches', type=int, default=8)
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=100,
                        help='score 计算使用的训练 batch 数')
    parser.add_argument('--eval_batches', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str,
                        default='results/paper_results/sensitivity')

    # 选择性运行
    parser.add_argument('--skip_k', action='store_true',
                        help='跳过 K sweep')
    parser.add_argument('--skip_hvp', action='store_true',
                        help='跳过 HVP batch size sweep')
    parser.add_argument('--skip_tol', action='store_true',
                        help='跳过 bisection tolerance sweep')
    parser.add_argument('--skip_timing', action='store_true',
                        help='跳过 timing overhead breakdown')
    args = parser.parse_args()

    print("=" * 70)
    print("敏感性分析")
    print(f"模型: {args.model} | 数据集: {args.dataset} | 剪枝比例: {args.prune_ratio}")
    print("=" * 70)

    # 加载
    model_init, model_family = load_model(args.model, pretrained=True,
                               checkpoint_path=args.checkpoint, device='cpu',
                               dataset_name=args.dataset)
    train_loader, val_loader, task_type = get_data_loaders(
        args.model, args.dataset, args.batch_size, args.seq_length)

    cached_train = cache_batches(train_loader, args.num_steps, task_type)
    cached_eval = cache_batches(val_loader, args.eval_batches, task_type)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"{timestamp}_{args.model}_{args.dataset}"

    all_results = {}

    # --- Sweep K ---
    if not args.skip_k:
        print("\n[1/4] Sweep K (compression-restore count)...")
        k_results = sweep_K(args, model_init, train_loader, val_loader,
                            task_type, cached_eval)
        all_results['sweep_K'] = k_results
        print_results_table(k_results, ['K', 'method', 'loss', 'perplexity'])

    # --- Sweep HVP batch size ---
    if not args.skip_hvp:
        print("\n[2/4] Sweep HVP batch size...")
        hvp_results = sweep_hvp_batch(args, model_init, cached_train,
                                       cached_eval, task_type,
                                       model_family=model_family)
        all_results['sweep_hvp_batches'] = hvp_results
        print_results_table(hvp_results, ['hvp_batches', 'hvp_time_s',
                                           'loss', 'perplexity'])

    # --- Sweep bisection tolerance ---
    if not args.skip_tol:
        print("\n[3/4] Sweep bisection tolerance...")
        tol_results = sweep_bisection_tol(args, model_init, cached_train,
                                           cached_eval, task_type,
                                           model_family=model_family)
        all_results['sweep_bisection_tol'] = tol_results
        print_results_table(tol_results, ['bisection_tol', 'alloc_time_s',
                                           'loss', 'perplexity'])

    # --- Timing overhead breakdown ---
    if not args.skip_timing:
        print("\n[4/4] Timing overhead breakdown...")
        timing_result = timing_breakdown(args, model_init, cached_train,
                                          cached_eval, task_type,
                                          model_family=model_family)
        all_results['timing_breakdown'] = timing_result

    # 保存
    with open(output_dir / f'{prefix}_sensitivity.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("完成！")


if __name__ == '__main__':
    main()
