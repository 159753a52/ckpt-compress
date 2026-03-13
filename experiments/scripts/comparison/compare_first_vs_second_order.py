"""
一阶 vs 一阶+二阶重要性方法对比实验。

关键特性：
- 使用完全相同的数据批次计算梯度和动量
- 使用完全相同的数据批次评估损失
- 同时测试多个剪枝比例
- 公平对比两种重要性计算方法

运行命令:
    python experiments/scripts/compare_first_vs_second_order.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --num_steps 100 \
        --prune_ratios 0.10,0.15,0.20,0.25,0.30 \
        --eval_batches 10 \
        --batch_size 4 \
        --seq_length 512 \
        --device cuda

重要性公式:
- 一阶: d_i = |g_i * θ_i|
- 一阶+二阶: d_i = |g_i * θ_i| + α * |v_i * θ_i²|
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from collections import defaultdict
from scipy import stats
from scipy.optimize import bisect
import pandas as pd
from datetime import datetime
import copy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader
from src.ckpt_compress.methods.adam_prune.importance import (
    compute_importance_scores_abs,
    compute_importance_scores_first_order,
)


def collect_gradients_and_momentum(model, cached_train_batches, device, num_steps=100):
    """使用缓存的训练批次累积梯度和动量。"""
    model.train()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    accumulated_gradients = defaultdict(lambda: 0)
    accumulated_exp_avg_sq = defaultdict(lambda: 0)

    print(f"累积 {num_steps} 步的梯度和动量...")

    num_batches = len(cached_train_batches)

    for step in tqdm(range(num_steps)):
        batch = cached_train_batches[step % num_batches]

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                accumulated_gradients[name] = accumulated_gradients[name] + param.grad.detach().cpu()

        optimizer.step()

        for name, param in model.named_parameters():
            if name in optimizer.state[param]:
                exp_avg_sq = optimizer.state[param]['exp_avg_sq']
                accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] + exp_avg_sq.detach().cpu()

    for name in accumulated_gradients:
        accumulated_gradients[name] = accumulated_gradients[name] / num_steps
        accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] / num_steps

    weights = {name: param.detach().cpu() for name, param in model.named_parameters()}
    return weights, dict(accumulated_gradients), dict(accumulated_exp_avg_sq)


def fit_gamma_distribution(data):
    """拟合 Gamma 分布，返回 (k, θ)。"""
    data_positive = data[data > 0]
    if len(data_positive) == 0:
        return None, None

    k, loc, scale = stats.gamma.fit(data_positive, floc=0)
    return k, scale


def solve_global_threshold(layer_info, global_prune_ratio):
    """求解全局阈值 τ*。"""
    N = sum(info['n_params'] for info in layer_info)

    def objective(tau):
        total_pruned = 0
        for info in layer_info:
            k, theta = info['k'], info['theta']
            if k is None or theta is None:
                continue
            cdf_value = stats.gamma.cdf(tau, k, scale=theta)
            total_pruned += info['n_params'] * cdf_value
        return total_pruned / N - global_prune_ratio

    all_scores = []
    for info in layer_info:
        if info['scores'] is not None:
            all_scores.extend(info['scores'].flatten().tolist())

    all_scores = np.array(all_scores)
    tau_min = np.percentile(all_scores, 0.1)
    tau_max = np.percentile(all_scores, 50)

    try:
        tau_star = bisect(objective, tau_min, tau_max, xtol=1e-10, maxiter=100)
    except ValueError:
        tau_max = np.percentile(all_scores, 90)
        try:
            tau_star = bisect(objective, tau_min, tau_max, xtol=1e-10, maxiter=100)
        except ValueError:
            tau_max = np.percentile(all_scores, 99)
            tau_star = bisect(objective, tau_min, tau_max, xtol=1e-10, maxiter=100)

    return tau_star


def compute_layer_prune_ratios(layer_info, tau_star):
    """计算各层剪枝率。"""
    for info in layer_info:
        k, theta = info['k'], info['theta']
        if k is None or theta is None:
            info['prune_ratio'] = 0.0
        else:
            info['prune_ratio'] = stats.gamma.cdf(tau_star, k, scale=theta)
    return layer_info


def apply_pruning_with_threshold(model, layer_info, tau_star):
    """应用剪枝。"""
    actual_pruned = 0
    total_params = 0

    for info in layer_info:
        name = info['name']
        scores = info['scores']
        if scores is None:
            continue

        mask = (scores >= tau_star).float()
        param = dict(model.named_parameters())[name]
        param.data.mul_(mask.to(param.device))

        actual_pruned += (mask == 0).sum().item()
        total_params += mask.numel()

    actual_ratio = actual_pruned / total_params if total_params > 0 else 0
    return model, actual_ratio


def cache_batches(data_loader, num_batches):
    """缓存数据批次。"""
    cached = []
    data_iter = iter(data_loader)
    for _ in range(num_batches):
        try:
            batch = next(data_iter)
            cached.append({
                'input_ids': batch['input_ids'].clone(),
                'labels': batch['labels'].clone()
            })
        except StopIteration:
            break
    return cached


def evaluate_loss_with_cached_batches(model, cached_batches, device):
    """使用缓存批次评估损失。"""
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    with torch.no_grad():
        for batch in cached_batches:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()

    return total_loss / len(cached_batches)


def build_layer_info(scores):
    """构建层信息并拟合 Gamma 分布。"""
    layer_info = []
    for name, score_tensor in scores.items():
        if 'weight' not in name:
            continue
        if 'wte.weight' in name or 'wpe.weight' in name:
            continue

        scores_flat = score_tensor.flatten().numpy()
        k, theta = fit_gamma_distribution(scores_flat)

        layer_info.append({
            'name': name,
            'n_params': len(scores_flat),
            'k': k,
            'theta': theta,
            'scores': score_tensor,
            'mean': np.mean(scores_flat),
            'median': np.median(scores_flat),
        })
    return layer_info


def run_pruning_for_ratio(checkpoint, layer_info, prune_ratio, cached_eval_batches, device):
    """对单个剪枝比例运行实验。"""
    tau_star = solve_global_threshold(layer_info, prune_ratio)
    layer_info_copy = copy.deepcopy(layer_info)
    layer_info_copy = compute_layer_prune_ratios(layer_info_copy, tau_star)

    model_pruned = get_gpt2_small(pretrained=False)
    model_pruned.load_state_dict(checkpoint['model_state_dict'])
    model_pruned, actual_ratio = apply_pruning_with_threshold(model_pruned, layer_info_copy, tau_star)

    loss_pruned = evaluate_loss_with_cached_batches(model_pruned, cached_eval_batches, device)

    prune_ratios_per_layer = [info['prune_ratio'] for info in layer_info_copy]

    return {
        'tau_star': tau_star,
        'loss_pruned': loss_pruned,
        'actual_ratio': actual_ratio,
        'layer_prune_ratios': prune_ratios_per_layer,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='一阶 vs 一阶+二阶重要性方法对比实验')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt')
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--prune_ratios', type=str, default='0.10,0.15,0.20,0.25,0.30')
    parser.add_argument('--eval_batches', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output_dir', type=str, default='results/compare_first_vs_second_order')

    args = parser.parse_args()

    prune_ratios = [float(x.strip()) for x in args.prune_ratios.split(',')]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("一阶 vs 一阶+二阶重要性方法对比实验")
    print("=" * 80)
    print(f"剪枝比例: {[f'{r*100:.0f}%' for r in prune_ratios]}")
    print(f"评估批次数: {args.eval_batches}")
    print(f"二阶项权重 α: {args.alpha}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)

    # 1. 加载模型
    print("\n[1/8] 加载模型...")
    model = get_gpt2_small(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ 模型已加载")

    # 2. 加载数据
    print("\n[2/8] 加载数据...")
    data_loader = get_wikitext103_dataloader(
        split='train',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0
    )

    # 3. 缓存训练批次（用于计算梯度）和评估批次
    print(f"\n[3/8] 缓存数据批次...")
    # 缓存足够多的训练批次
    num_train_batches = max(args.num_steps, 200)
    cached_train_batches = cache_batches(data_loader, num_train_batches)
    print(f"  ✓ 已缓存 {len(cached_train_batches)} 个训练批次")

    # 重新创建 data_loader 以获取不同的评估批次
    data_loader_eval = get_wikitext103_dataloader(
        split='validation',  # 使用验证集评估
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0
    )
    cached_eval_batches = cache_batches(data_loader_eval, args.eval_batches)
    print(f"  ✓ 已缓存 {len(cached_eval_batches)} 个评估批次（验证集）")

    # 4. 计算梯度和动量（只计算一次，两种方法共用）
    print("\n[4/8] 计算梯度和动量（共用）...")
    weights, gradients, exp_avg_sq = collect_gradients_and_momentum(
        model, cached_train_batches, device, num_steps=args.num_steps
    )
    print("✓ 梯度和动量已计算")

    # 5. 计算两种重要性得分
    print("\n[5/8] 计算重要性得分...")

    # 一阶
    scores_first_order = compute_importance_scores_first_order(
        weights=weights,
        gradients=gradients,
        exp_avg_sq=exp_avg_sq,
        alpha=args.alpha
    )
    print("  ✓ 一阶重要性得分已计算: d_i = |g_i * θ_i|")

    # 一阶+二阶
    scores_second_order = compute_importance_scores_abs(
        weights=weights,
        gradients=gradients,
        exp_avg_sq=exp_avg_sq,
        alpha=args.alpha
    )
    print(f"  ✓ 一阶+二阶重要性得分已计算: d_i = |g_i * θ_i| + {args.alpha} * |v_i * θ_i²|")

    # 6. 为每层拟合 Gamma 分布
    print("\n[6/8] 为每层拟合 Gamma 分布...")
    layer_info_first = build_layer_info(scores_first_order)
    layer_info_second = build_layer_info(scores_second_order)
    print(f"  ✓ 已拟合 {len(layer_info_first)} 层")

    # 7. 评估原始模型损失
    print("\n[7/8] 评估原始模型...")
    model_original = get_gpt2_small(pretrained=False)
    model_original.load_state_dict(checkpoint['model_state_dict'])
    loss_original = evaluate_loss_with_cached_batches(model_original, cached_eval_batches, device)
    print(f"✓ 原始损失: {loss_original:.4f}")

    # 8. 对每个剪枝比例运行两种方法
    print("\n[8/8] 运行对比实验...")
    results_first = []
    results_second = []

    for prune_ratio in prune_ratios:
        print(f"\n--- 剪枝比例: {prune_ratio*100:.0f}% ---")

        # 一阶方法
        result_first = run_pruning_for_ratio(
            checkpoint, layer_info_first, prune_ratio, cached_eval_batches, device
        )
        loss_inc_first = result_first['loss_pruned'] - loss_original
        loss_inc_pct_first = (loss_inc_first / loss_original) * 100

        results_first.append({
            'target_ratio': prune_ratio,
            'actual_ratio': result_first['actual_ratio'],
            'tau_star': result_first['tau_star'],
            'loss_original': loss_original,
            'loss_pruned': result_first['loss_pruned'],
            'loss_increase': loss_inc_first,
            'loss_increase_pct': loss_inc_pct_first,
        })

        # 一阶+二阶方法
        result_second = run_pruning_for_ratio(
            checkpoint, layer_info_second, prune_ratio, cached_eval_batches, device
        )
        loss_inc_second = result_second['loss_pruned'] - loss_original
        loss_inc_pct_second = (loss_inc_second / loss_original) * 100

        results_second.append({
            'target_ratio': prune_ratio,
            'actual_ratio': result_second['actual_ratio'],
            'tau_star': result_second['tau_star'],
            'loss_original': loss_original,
            'loss_pruned': result_second['loss_pruned'],
            'loss_increase': loss_inc_second,
            'loss_increase_pct': loss_inc_pct_second,
        })

        print(f"  一阶:       实际率={result_first['actual_ratio']*100:.2f}%, "
              f"损失={result_first['loss_pruned']:.4f}, 增加={loss_inc_pct_first:+.2f}%")
        print(f"  一阶+二阶:  实际率={result_second['actual_ratio']*100:.2f}%, "
              f"损失={result_second['loss_pruned']:.4f}, 增加={loss_inc_pct_second:+.2f}%")

    # 汇总结果
    print("\n" + "=" * 90)
    print("实验结果汇总（使用相同数据）")
    print("=" * 90)
    print(f"原始损失: {loss_original:.4f}")
    print()
    print(f"{'剪枝率':<8} {'一阶损失':<12} {'一阶增加%':<12} {'二阶损失':<12} {'二阶增加%':<12} {'差异':<10}")
    print("-" * 80)

    comparison_results = []
    for r1, r2 in zip(results_first, results_second):
        diff = r1['loss_increase_pct'] - r2['loss_increase_pct']
        better = "一阶更优" if diff < 0 else "二阶更优" if diff > 0 else "相同"

        print(f"{r1['target_ratio']*100:>5.0f}%   "
              f"{r1['loss_pruned']:<12.4f} "
              f"{r1['loss_increase_pct']:>+10.2f}%  "
              f"{r2['loss_pruned']:<12.4f} "
              f"{r2['loss_increase_pct']:>+10.2f}%  "
              f"{diff:>+8.2f}% ({better})")

        comparison_results.append({
            'target_ratio': r1['target_ratio'],
            'first_order_loss': r1['loss_pruned'],
            'first_order_increase_pct': r1['loss_increase_pct'],
            'second_order_loss': r2['loss_pruned'],
            'second_order_increase_pct': r2['loss_increase_pct'],
            'difference_pct': diff,
            'better_method': better,
        })

    # 保存结果
    pd.DataFrame(results_first).to_csv(output_dir / 'results_first_order.csv', index=False)
    pd.DataFrame(results_second).to_csv(output_dir / 'results_second_order.csv', index=False)
    pd.DataFrame(comparison_results).to_csv(output_dir / 'comparison.csv', index=False)

    # 保存配置
    config = {
        'checkpoint': args.checkpoint,
        'num_steps': args.num_steps,
        'prune_ratios': prune_ratios,
        'eval_batches': args.eval_batches,
        'batch_size': args.batch_size,
        'seq_length': args.seq_length,
        'alpha': args.alpha,
        'device': device,
        'loss_original': loss_original,
        'first_order_formula': 'd_i = |g_i * θ_i|',
        'second_order_formula': f'd_i = |g_i * θ_i| + {args.alpha} * |v_i * θ_i²|',
    }

    with open(output_dir / 'config.txt', 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    print(f"\n✓ 结果已保存到: {output_dir}")
    print("=" * 90)


if __name__ == '__main__':
    main()
