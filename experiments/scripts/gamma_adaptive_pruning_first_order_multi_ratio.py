"""
基于 Gamma 分布的自适应剪枝实验（纯一阶重要性）- 多剪枝比例版本。

运行命令:
    python experiments/scripts/gamma_adaptive_pruning_first_order_multi_ratio.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --num_steps 100 \
        --prune_ratios 0.10,0.15,0.20,0.25,0.30 \
        --eval_batches 10 \
        --batch_size 4 \
        --seq_length 512 \
        --device cuda

重要性公式: d_i = |g_i * θ_i| (纯一阶)

与标准版本的区别:
- 标准版本: d_i = |g_i * θ_i| + α * |v_i * θ_i²|
- 本版本: d_i = |g_i * θ_i| (纯一阶)

关键改进:
- 重要性得分和 Gamma 分布拟合只计算一次
- 评估数据批次预先缓存，确保一致性
- 支持多个剪枝比例的批量测试
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
from src.ckpt_compress.methods.adam_prune.importance import compute_importance_scores_first_order


def collect_gradients_and_momentum(model, data_loader, device, num_steps=100):
    """累积梯度和动量。"""
    model.train()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    accumulated_gradients = defaultdict(lambda: 0)
    accumulated_exp_avg_sq = defaultdict(lambda: 0)

    print(f"累积 {num_steps} 步的梯度和动量...")
    data_iter = iter(data_loader)

    for step in tqdm(range(num_steps)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

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
    """求解全局阈值 τ*，使得全局剪枝率 = global_prune_ratio。"""
    N = sum(info['n_params'] for info in layer_info)

    def objective(tau):
        total_pruned = 0
        for info in layer_info:
            k, theta = info['k'], info['theta']
            if k is None or theta is None:
                continue
            cdf_value = stats.gamma.cdf(tau, k, scale=theta)
            total_pruned += info['n_params'] * cdf_value

        current_ratio = total_pruned / N
        return current_ratio - global_prune_ratio

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
    """计算各层剪枝率: p_ℓ = P(k_ℓ, τ*/θ_ℓ)"""
    for info in layer_info:
        k, theta = info['k'], info['theta']
        if k is None or theta is None:
            info['prune_ratio'] = 0.0
        else:
            info['prune_ratio'] = stats.gamma.cdf(tau_star, k, scale=theta)

    return layer_info


def apply_pruning_with_threshold(model, layer_info, tau_star):
    """应用剪枝：将重要性 < τ* 的参数置零。"""
    mask_dict = {}
    actual_pruned = 0
    total_params = 0

    for info in layer_info:
        name = info['name']
        scores = info['scores']

        if scores is None:
            continue

        mask = (scores >= tau_star).float()
        mask_dict[name] = mask

        param = dict(model.named_parameters())[name]
        param.data.mul_(mask.to(param.device))

        actual_pruned += (mask == 0).sum().item()
        total_params += mask.numel()

    actual_ratio = actual_pruned / total_params if total_params > 0 else 0
    return model, mask_dict, actual_ratio


def cache_eval_batches(data_loader, num_batches):
    """缓存评估数据批次，确保所有实验使用相同数据。"""
    cached_batches = []
    data_iter = iter(data_loader)

    for _ in range(num_batches):
        try:
            batch = next(data_iter)
            cached_batches.append({
                'input_ids': batch['input_ids'].clone(),
                'labels': batch['labels'].clone()
            })
        except StopIteration:
            break

    return cached_batches


def evaluate_loss_with_cached_batches(model, cached_batches, device):
    """使用缓存的批次评估模型损失。"""
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


def run_pruning_experiment(checkpoint, layer_info, prune_ratio, cached_batches, device):
    """对单个剪枝比例运行实验。"""
    # 求解全局阈值
    tau_star = solve_global_threshold(layer_info, prune_ratio)

    # 计算各层剪枝率
    layer_info_copy = copy.deepcopy(layer_info)
    layer_info_copy = compute_layer_prune_ratios(layer_info_copy, tau_star)

    # 加载新模型并应用剪枝
    model_pruned = get_gpt2_small(pretrained=False)
    model_pruned.load_state_dict(checkpoint['model_state_dict'])
    model_pruned, mask_dict, actual_ratio = apply_pruning_with_threshold(
        model_pruned, layer_info_copy, tau_star
    )

    # 评估剪枝后损失
    loss_pruned = evaluate_loss_with_cached_batches(model_pruned, cached_batches, device)

    # 收集各层剪枝率统计
    prune_ratios_per_layer = [info['prune_ratio'] for info in layer_info_copy]

    return {
        'tau_star': tau_star,
        'loss_pruned': loss_pruned,
        'actual_ratio': actual_ratio,
        'layer_prune_ratios': prune_ratios_per_layer,
        'layer_info': layer_info_copy,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='基于 Gamma 分布的自适应剪枝实验（纯一阶）- 多剪枝比例版本')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt',
                        help='检查点路径')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='累积的训练步数')
    parser.add_argument('--prune_ratios', type=str, default='0.10,0.15,0.20,0.25,0.30',
                        help='剪枝比例列表，逗号分隔')
    parser.add_argument('--eval_batches', type=int, default=10,
                        help='评估损失的批次数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='序列长度')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备')
    parser.add_argument('--output_dir', type=str,
                        default='results/gamma_adaptive_pruning_first_order_multi',
                        help='输出目录')

    args = parser.parse_args()

    # 解析剪枝比例
    prune_ratios = [float(x.strip()) for x in args.prune_ratios.split(',')]

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("基于 Gamma 分布的自适应剪枝实验（纯一阶）- 多剪枝比例版本")
    print("=" * 80)
    print(f"重要性公式: d_i = |g_i * θ_i| (纯一阶)")
    print(f"剪枝比例: {[f'{r*100:.0f}%' for r in prune_ratios]}")
    print(f"评估批次数: {args.eval_batches}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)

    # 1. 加载模型
    print("\n[1/7] 加载模型...")
    model = get_gpt2_small(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ 模型已加载")

    # 2. 加载数据
    print("\n[2/7] 加载数据...")
    data_loader = get_wikitext103_dataloader(
        split='train',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0
    )

    # 3. 缓存评估数据批次
    print(f"\n[3/7] 缓存 {args.eval_batches} 个评估批次...")
    cached_batches = cache_eval_batches(data_loader, args.eval_batches)
    print(f"✓ 已缓存 {len(cached_batches)} 个批次")

    # 4. 计算重要性得分（纯一阶，只计算一次）
    print("\n[4/7] 计算参数重要性（纯一阶）...")
    weights, gradients, exp_avg_sq = collect_gradients_and_momentum(
        model, data_loader, device, num_steps=args.num_steps
    )

    scores = compute_importance_scores_first_order(
        weights=weights,
        gradients=gradients,
        exp_avg_sq=exp_avg_sq,
        alpha=0.5  # 未使用，保持接口一致
    )
    print("✓ 重要性得分已计算（纯一阶）")

    # 5. 为每层拟合 Gamma 分布（只拟合一次）
    print("\n[5/7] 为每层拟合 Gamma 分布（排除 Embedding 层）...")
    layer_info = []

    for name, score_tensor in tqdm(scores.items()):
        if 'weight' not in name:
            continue

        if 'wte.weight' in name or 'wpe.weight' in name:
            print(f"  跳过 Embedding 层: {name}")
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

    print(f"✓ 已拟合 {len(layer_info)} 层")

    # 6. 评估原始模型损失（使用缓存批次）
    print("\n[6/7] 评估原始模型...")
    model_original = get_gpt2_small(pretrained=False)
    model_original.load_state_dict(checkpoint['model_state_dict'])
    loss_original = evaluate_loss_with_cached_batches(model_original, cached_batches, device)
    print(f"✓ 原始损失: {loss_original:.4f}")

    # 7. 对每个剪枝比例运行实验
    print("\n[7/7] 运行多剪枝比例实验...")
    results = []

    for prune_ratio in prune_ratios:
        print(f"\n--- 剪枝比例: {prune_ratio*100:.0f}% ---")

        result = run_pruning_experiment(
            checkpoint=checkpoint,
            layer_info=layer_info,
            prune_ratio=prune_ratio,
            cached_batches=cached_batches,
            device=device
        )

        loss_increase = result['loss_pruned'] - loss_original
        loss_increase_pct = (loss_increase / loss_original) * 100

        results.append({
            'target_ratio': prune_ratio,
            'actual_ratio': result['actual_ratio'],
            'tau_star': result['tau_star'],
            'loss_original': loss_original,
            'loss_pruned': result['loss_pruned'],
            'loss_increase': loss_increase,
            'loss_increase_pct': loss_increase_pct,
            'layer_prune_ratio_min': min(result['layer_prune_ratios']),
            'layer_prune_ratio_max': max(result['layer_prune_ratios']),
            'layer_prune_ratio_mean': np.mean(result['layer_prune_ratios']),
            'layer_prune_ratio_median': np.median(result['layer_prune_ratios']),
        })

        print(f"  全局阈值 τ*: {result['tau_star']:.6e}")
        print(f"  实际剪枝率: {result['actual_ratio']*100:.2f}%")
        print(f"  剪枝后损失: {result['loss_pruned']:.4f}")
        print(f"  损失增加: {loss_increase:.4f} ({loss_increase_pct:+.2f}%)")

        # 保存各层剪枝率
        layer_df = pd.DataFrame(result['layer_info'])
        layer_df = layer_df[['name', 'n_params', 'k', 'theta', 'mean', 'median', 'prune_ratio']]
        layer_df.to_csv(output_dir / f'layer_prune_ratios_{int(prune_ratio*100)}pct.csv', index=False)

    # 汇总结果
    print("\n" + "=" * 80)
    print("实验结果汇总（纯一阶重要性）")
    print("=" * 80)
    print(f"{'剪枝率':<10} {'实际率':<10} {'原始损失':<12} {'剪枝损失':<12} {'损失增加':<12} {'增加%':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['target_ratio']*100:>6.0f}%    "
              f"{r['actual_ratio']*100:>6.2f}%    "
              f"{r['loss_original']:<12.4f} "
              f"{r['loss_pruned']:<12.4f} "
              f"{r['loss_increase']:<12.4f} "
              f"{r['loss_increase_pct']:>+8.2f}%")

    # 保存汇总结果
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / 'summary.csv', index=False)
    print(f"\n✓ 汇总结果已保存到: {output_dir / 'summary.csv'}")

    # 保存详细配置
    config = {
        'checkpoint': args.checkpoint,
        'num_steps': args.num_steps,
        'prune_ratios': prune_ratios,
        'eval_batches': args.eval_batches,
        'batch_size': args.batch_size,
        'seq_length': args.seq_length,
        'importance_formula': 'd_i = |g_i * θ_i| (first-order only)',
        'device': device,
        'num_layers': len(layer_info),
        'loss_original': loss_original,
    }

    with open(output_dir / 'config.txt', 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    print(f"✓ 配置已保存到: {output_dir / 'config.txt'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
