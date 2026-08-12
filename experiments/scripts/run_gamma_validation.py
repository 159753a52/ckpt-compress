"""
9.1 Gamma 近似精度验证 + 9.2 多分布对比

验证 Gamma CDF 近似的精确度，并与其他分布进行比较。

功能:
  1. KS 检验每层 Gamma 拟合质量
  2. 对比 Empirical CDF / Gamma CDF / Uniform 的下游质量差异
  3. 对比 Gamma / LogNormal / Weibull / Exponential 拟合质量

运行示例:
    python experiments/scripts/run_gamma_validation.py \
        --model gpt2-medium --dataset wikitext103 \
        --prune_ratios 0.2,0.3,0.4 \
        --num_steps 100 --eval_batches 20 \
        --device cuda
"""

import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

import sys
from pathlib import Path
import argparse
import copy
import json
import torch
import numpy as np
from scipy import stats
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lib.models import load_model
from experiments.lib.data import get_data_loaders, cache_batches
from experiments.lib.evaluation import evaluate
from experiments.lib.results import save_results, print_results_table
from experiments.lib.importance_compare.scoring import compute_scores_by_method
from dacp.pruning import apply_pruning
from dacp.pruning.allocation import (
    UniformAllocation,
    GammaAdaptiveAllocation,
    GlobalTopKAllocation,
    WeibullAdaptiveAllocation,
)


# ============================================================
#  Part 1: KS 检验 + 多分布对比
# ============================================================

def fit_and_test_distributions(scores_dict):
    """对每层拟合多种分布，返回拟合质量指标。"""
    distributions = {
        'gamma': lambda data: _fit_gamma(data),
        'lognormal': lambda data: _fit_lognorm(data),
        'weibull': lambda data: _fit_weibull(data),
        'exponential': lambda data: _fit_exponential(data),
    }

    MAX_SAMPLES = 50000  # 子采样加速拟合
    rng = np.random.RandomState(42)

    results = []
    for name, score in scores_dict.items():
        data = score.flatten().float().cpu().numpy()
        data_pos = data[data > 0]
        if len(data_pos) < 50:
            continue
        # 子采样
        if len(data_pos) > MAX_SAMPLES:
            data_pos = rng.choice(data_pos, MAX_SAMPLES, replace=False)

        row = {
            'layer': name,
            'n_params': len(data),
            'n_positive': len(data_pos),
            'mean': float(data_pos.mean()),
            'std': float(data_pos.std()),
        }

        for dist_name, fit_fn in distributions.items():
            try:
                ks_stat, p_value, params = fit_fn(data_pos)
                row[f'{dist_name}_ks'] = float(ks_stat)
                row[f'{dist_name}_pvalue'] = float(p_value)
                row[f'{dist_name}_params'] = [float(value) for value in params]
                # AIC
                if dist_name == 'gamma':
                    ll = np.sum(stats.gamma.logpdf(data_pos, *params))
                elif dist_name == 'lognormal':
                    ll = np.sum(stats.lognorm.logpdf(data_pos, *params))
                elif dist_name == 'weibull':
                    ll = np.sum(stats.weibull_min.logpdf(data_pos, *params))
                elif dist_name == 'exponential':
                    ll = np.sum(stats.expon.logpdf(data_pos, *params))
                k_params = len(params) - 1  # exclude loc
                aic = 2 * k_params - 2 * ll
                row[f'{dist_name}_aic'] = float(aic) if np.isfinite(aic) else None
            except Exception as e:
                row[f'{dist_name}_ks'] = None
                row[f'{dist_name}_pvalue'] = 0.0
                row[f'{dist_name}_aic'] = None

        cdf_x = np.unique(np.quantile(data_pos, np.linspace(0.001, 0.999, 256)))
        sorted_data = np.sort(data_pos)
        row['cdf'] = {
            'x': cdf_x.tolist(),
            'empirical': (
                np.searchsorted(sorted_data, cdf_x, side='right') / len(sorted_data)
            ).tolist(),
        }
        fitted_cdfs = {
            'gamma': stats.gamma,
            'lognormal': stats.lognorm,
            'weibull': stats.weibull_min,
            'exponential': stats.expon,
        }
        for dist_name, distribution in fitted_cdfs.items():
            params = row.get(f'{dist_name}_params')
            if params is not None:
                row['cdf'][dist_name] = distribution.cdf(cdf_x, *params).tolist()
        results.append(row)
    return results


def _fit_gamma(data):
    # 矩估计（O(N)，替代 MLE 迭代）
    mean_x = data.mean()
    var_x = data.var()
    if var_x < 1e-30:
        var_x = 1e-30
    k = mean_x ** 2 / var_x  # shape
    theta = var_x / mean_x   # scale
    params = (k, 0, theta)    # (a, loc, scale)
    ks_stat, p_value = stats.kstest(data, 'gamma', args=params)
    return ks_stat, p_value, params


def _fit_lognorm(data):
    params = stats.lognorm.fit(data, floc=0)
    ks_stat, p_value = stats.kstest(data, 'lognorm', args=params)
    return ks_stat, p_value, params


def _fit_weibull(data):
    params = stats.weibull_min.fit(data, floc=0)
    ks_stat, p_value = stats.kstest(data, 'weibull_min', args=params)
    return ks_stat, p_value, params


def _fit_exponential(data):
    params = stats.expon.fit(data, floc=0)
    ks_stat, p_value = stats.kstest(data, 'expon', args=params)
    return ks_stat, p_value, params


# ============================================================
#  Part 2: Empirical CDF / Gamma CDF / Uniform 质量对比
# ============================================================

def run_allocation_comparison(model_factory, scores, prune_ratios,
                              cached_eval, task_type, device, baseline):
    """对比三种分配策略的下游质量。"""
    strategies = {
        'uniform': UniformAllocation(),
        'gamma-adaptive': GammaAdaptiveAllocation(),
        'global-topk': GlobalTopKAllocation(),  # Empirical CDF 精确版
        'weibull-adaptive': WeibullAdaptiveAllocation(),
    }

    results = []
    for strat_name, allocator in strategies.items():
        for ratio in prune_ratios:
            model_copy = model_factory().to(device)
            layer_ratios = allocator.allocate(scores, ratio)
            _, _, actual = apply_pruning(model_copy, scores, layer_ratios, device=device)
            metrics = evaluate(model_copy, cached_eval, task_type, device)

            result = {
                'allocation': strat_name,
                'target_ratio': ratio,
                'actual_ratio': actual,
            }
            result.update(metrics)
            if 'loss' in baseline:
                result['loss_increase_pct'] = (
                    (metrics['loss'] - baseline['loss']) / baseline['loss'] * 100
                )
            results.append(result)

            print(f"  {strat_name} | ratio={ratio:.0%} (actual={actual:.1%}) | "
                  + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            del model_copy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description='Gamma 近似验证 + 多分布对比')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--importance', type=str, default='second-order-hvp',
                        help='用于计算 score 的方法')
    parser.add_argument('--prune_ratios', type=str, default='0.2,0.3,0.4')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--hvp_batches', type=int, default=8)
    parser.add_argument('--hvp_mode', type=str, default='full')
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--eval_batches', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str,
                        default='results/paper_results/gamma_validation')
    args = parser.parse_args()

    prune_ratios = [float(r) for r in args.prune_ratios.split(',')]

    print("=" * 70)
    print("Gamma 近似验证 + 多分布对比")
    print(f"模型: {args.model} | 数据集: {args.dataset}")
    print(f"Importance: {args.importance} | alpha: {args.alpha}")
    print("=" * 70)

    # 加载
    model, model_family = load_model(args.model, pretrained=True,
                          checkpoint_path=args.checkpoint, device='cpu',
                          dataset_name=args.dataset)
    train_loader, val_loader, task_type = get_data_loaders(
        args.model, args.dataset, args.batch_size, args.seq_length,
        data_dir=args.data_dir)

    cached_train = cache_batches(train_loader, args.num_steps, task_type)
    cached_eval = cache_batches(val_loader, args.eval_batches, task_type)

    # Baseline
    model_eval = copy.deepcopy(model).to(args.device)
    baseline = evaluate(model_eval, cached_eval, task_type, args.device)
    print(f"Baseline: {baseline}")
    del model_eval

    # 计算 importance scores
    print("\n[1/3] 计算 importance scores...")
    model_for_scoring = copy.deepcopy(model).to(args.device)
    score_cache = compute_scores_by_method(
        model_for_scoring, cached_train, task_type,
        methods=[args.importance], alpha=args.alpha,
        hvp_batches=args.hvp_batches, hvp_mode=args.hvp_mode,
        chunk_size=args.chunk_size,
        model_family=model_family,
    )
    scores = score_cache[args.importance]
    del model_for_scoring
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Part 1: 分布拟合质量
    print("\n[2/3] 分布拟合质量 (KS 检验)...")
    fit_results = fit_and_test_distributions(scores)

    # 统计 Gamma 拟合通过率
    gamma_pass = sum(1 for r in fit_results if r.get('gamma_pvalue', 0) > 0.05)
    total_layers = len(fit_results)
    print(f"  Gamma KS pass rate (p>0.05): {gamma_pass}/{total_layers} "
          f"= {gamma_pass/max(total_layers,1)*100:.1f}%")

    # 每种分布的平均 KS-D
    for dist in ['gamma', 'lognormal', 'weibull', 'exponential']:
        ks_vals = [r[f'{dist}_ks'] for r in fit_results
                   if r.get(f'{dist}_ks') is not None]
        if ks_vals:
            print(f"  {dist:12s} avg KS-D = {np.mean(ks_vals):.4f}")

    # Part 2: 分配策略质量对比
    print("\n[3/3] 分配策略质量对比...")
    def model_factory():
        return copy.deepcopy(model)

    alloc_results = run_allocation_comparison(
        model_factory, scores, prune_ratios,
        cached_eval, task_type, args.device, baseline)

    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"{timestamp}_{args.model}_{args.dataset}"

    # 拟合结果
    with open(output_dir / f'{prefix}_fit_results.json', 'w') as f:
        json.dump({
            'schema_version': 1,
            'status': 'complete',
            'config': vars(args),
            'records': fit_results,
        }, f, indent=2, default=str, allow_nan=False)

    # 分配策略对比
    save_results(alloc_results, str(output_dir),
                 f"gamma_validation_{args.model}_{args.dataset}",
                 vars(args))

    print("\n" + "=" * 70)
    print("分配策略对比:")
    print_results_table(alloc_results, ['allocation', 'target_ratio',
                                         'actual_ratio', 'loss', 'perplexity'])
    print("\n完成！")


if __name__ == '__main__':
    main()
