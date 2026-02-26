"""
一阶 vs 一阶+二阶重要性方法对比实验 - 直接阈值版本。

关键改进：
- 不使用 Gamma 分布拟合，直接使用百分位数作为阈值
- 这样可以更准确地比较两种重要性计算方法的差异

运行命令:
    python experiments/scripts/compare_first_vs_second_order_direct.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --num_steps 100 \
        --prune_ratios 0.10,0.15,0.20,0.25,0.30 \
        --eval_batches 10 \
        --batch_size 4 \
        --seq_length 512 \
        --device cuda
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from datetime import datetime

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


def get_prunable_scores(scores):
    """获取可剪枝层的得分（排除 Embedding 和 bias）。"""
    prunable = {}
    for name, score_tensor in scores.items():
        if 'weight' not in name:
            continue
        if 'wte.weight' in name or 'wpe.weight' in name:
            continue
        prunable[name] = score_tensor.clone()  # 克隆以避免引用问题
    return prunable


def apply_global_pruning(model, scores, prune_ratio):
    """
    使用全局阈值进行剪枝。
    直接使用百分位数作为阈值，不经过分布拟合。
    """
    # 收集所有可剪枝参数的得分
    all_scores = []
    for name, score_tensor in scores.items():
        all_scores.append(score_tensor.flatten())

    all_scores = torch.cat(all_scores)

    # 使用 numpy 计算阈值（避免 torch.quantile 对大张量的限制）
    threshold = np.percentile(all_scores.numpy(), prune_ratio * 100)

    # 应用剪枝
    actual_pruned = 0
    total_params = 0

    for name, score_tensor in scores.items():
        mask = (score_tensor >= threshold).float()

        param = dict(model.named_parameters())[name]
        param.data.mul_(mask.to(param.device))

        actual_pruned += (mask == 0).sum().item()
        total_params += mask.numel()

    actual_ratio = actual_pruned / total_params
    return model, actual_ratio, threshold


def compute_mask_overlap(scores1, scores2, prune_ratio):
    """
    计算两种方法剪枝掩码的重叠率。
    """
    # 收集所有得分
    all_scores1 = torch.cat([s.flatten() for s in scores1.values()])
    all_scores2 = torch.cat([s.flatten() for s in scores2.values()])

    # 使用 numpy 计算百分位数（避免 torch.quantile 对大张量的限制）
    scores1_np = all_scores1.numpy()
    scores2_np = all_scores2.numpy()
    threshold1 = np.percentile(scores1_np, prune_ratio * 100)
    threshold2 = np.percentile(scores2_np, prune_ratio * 100)

    # 计算掩码
    mask1 = all_scores1.numpy() < threshold1  # 被剪枝的参数
    mask2 = all_scores2.numpy() < threshold2

    # 计算重叠
    both_pruned = (mask1 & mask2).sum()
    either_pruned = (mask1 | mask2).sum()
    only_first = (mask1 & ~mask2).sum()
    only_second = (~mask1 & mask2).sum()

    total = len(scores1_np)
    pruned1 = mask1.sum()
    pruned2 = mask2.sum()

    return {
        'total_params': total,
        'pruned_by_first': pruned1,
        'pruned_by_second': pruned2,
        'both_pruned': both_pruned,
        'only_first_pruned': only_first,
        'only_second_pruned': only_second,
        'overlap_ratio': both_pruned / either_pruned if either_pruned > 0 else 1.0,
        'jaccard_index': both_pruned / either_pruned if either_pruned > 0 else 1.0,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='一阶 vs 一阶+二阶重要性方法对比（直接阈值）')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt')
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--prune_ratios', type=str, default='0.10,0.15,0.20,0.25,0.30')
    parser.add_argument('--eval_batches', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output_dir', type=str, default='results/compare_first_vs_second_order_direct')

    args = parser.parse_args()

    prune_ratios = [float(x.strip()) for x in args.prune_ratios.split(',')]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 90)
    print("一阶 vs 一阶+二阶重要性方法对比实验（直接阈值版本）")
    print("=" * 90)
    print(f"剪枝比例: {[f'{r*100:.0f}%' for r in prune_ratios]}")
    print(f"二阶项权重 α: {args.alpha}")
    print(f"输出目录: {output_dir}")
    print("=" * 90)

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

    # 3. 缓存数据批次
    print(f"\n[3/7] 缓存数据批次...")
    num_train_batches = max(args.num_steps, 200)
    cached_train_batches = cache_batches(data_loader, num_train_batches)
    print(f"  ✓ 已缓存 {len(cached_train_batches)} 个训练批次")

    data_loader_eval = get_wikitext103_dataloader(
        split='validation',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0
    )
    cached_eval_batches = cache_batches(data_loader_eval, args.eval_batches)
    print(f"  ✓ 已缓存 {len(cached_eval_batches)} 个评估批次")

    # 4. 计算梯度和动量
    print("\n[4/7] 计算梯度和动量...")
    weights, gradients, exp_avg_sq = collect_gradients_and_momentum(
        model, cached_train_batches, device, num_steps=args.num_steps
    )
    print("✓ 梯度和动量已计算")

    # 5. 计算两种重要性得分
    print("\n[5/7] 计算重要性得分...")

    scores_first_order = compute_importance_scores_first_order(
        weights=weights, gradients=gradients, exp_avg_sq=exp_avg_sq, alpha=args.alpha
    )
    scores_first_order = get_prunable_scores(scores_first_order)
    print("  ✓ 一阶: d_i = |g_i * θ_i|")

    scores_second_order = compute_importance_scores_abs(
        weights=weights, gradients=gradients, exp_avg_sq=exp_avg_sq, alpha=args.alpha
    )
    scores_second_order = get_prunable_scores(scores_second_order)
    print(f"  ✓ 一阶+二阶: d_i = |g_i * θ_i| + {args.alpha} * |v_i * θ_i²|")

    # 分析得分差异
    print("\n[6/7] 分析重要性得分差异...")
    all_first = torch.cat([s.flatten() for s in scores_first_order.values()]).numpy()
    all_second = torch.cat([s.flatten() for s in scores_second_order.values()]).numpy()
    diff = all_second - all_first

    print(f"  一阶得分: mean={np.mean(all_first):.6e}, std={np.std(all_first):.6e}")
    print(f"  二阶得分: mean={np.mean(all_second):.6e}, std={np.std(all_second):.6e}")
    print(f"  差异:     mean={np.mean(diff):.6e}, std={np.std(diff):.6e}")
    print(f"  二阶项贡献: {np.mean(diff) / np.mean(all_second) * 100:.2f}%")

    # 6. 评估原始模型
    print("\n[7/7] 运行对比实验...")
    model_original = get_gpt2_small(pretrained=False)
    model_original.load_state_dict(checkpoint['model_state_dict'])
    loss_original = evaluate_loss_with_cached_batches(model_original, cached_eval_batches, device)
    print(f"原始损失: {loss_original:.4f}")

    # 7. 对每个剪枝比例运行实验
    results = []

    for prune_ratio in prune_ratios:
        print(f"\n--- 剪枝比例: {prune_ratio*100:.0f}% ---")

        # 分析掩码重叠
        overlap = compute_mask_overlap(scores_first_order, scores_second_order, prune_ratio)
        print(f"  掩码重叠率 (Jaccard): {overlap['jaccard_index']*100:.2f}%")
        print(f"  仅一阶剪枝: {overlap['only_first_pruned']:,}, 仅二阶剪枝: {overlap['only_second_pruned']:,}")

        # 一阶方法
        model_first = get_gpt2_small(pretrained=False)
        model_first.load_state_dict(checkpoint['model_state_dict'])
        model_first, actual_ratio_first, threshold_first = apply_global_pruning(
            model_first, scores_first_order, prune_ratio
        )
        loss_first = evaluate_loss_with_cached_batches(model_first, cached_eval_batches, device)
        loss_inc_first = (loss_first - loss_original) / loss_original * 100

        # 一阶+二阶方法
        model_second = get_gpt2_small(pretrained=False)
        model_second.load_state_dict(checkpoint['model_state_dict'])
        model_second, actual_ratio_second, threshold_second = apply_global_pruning(
            model_second, scores_second_order, prune_ratio
        )
        loss_second = evaluate_loss_with_cached_batches(model_second, cached_eval_batches, device)
        loss_inc_second = (loss_second - loss_original) / loss_original * 100

        print(f"  一阶:      损失={loss_first:.4f}, 增加={loss_inc_first:+.2f}%")
        print(f"  一阶+二阶: 损失={loss_second:.4f}, 增加={loss_inc_second:+.2f}%")

        diff_pct = loss_inc_first - loss_inc_second
        better = "一阶更优" if diff_pct < 0 else "二阶更优" if diff_pct > 0 else "相同"

        results.append({
            'target_ratio': prune_ratio,
            'actual_ratio_first': actual_ratio_first,
            'actual_ratio_second': actual_ratio_second,
            'threshold_first': threshold_first,
            'threshold_second': threshold_second,
            'loss_original': loss_original,
            'loss_first': loss_first,
            'loss_second': loss_second,
            'loss_inc_first_pct': loss_inc_first,
            'loss_inc_second_pct': loss_inc_second,
            'difference_pct': diff_pct,
            'better_method': better,
            'mask_overlap_jaccard': overlap['jaccard_index'],
            'only_first_pruned': overlap['only_first_pruned'],
            'only_second_pruned': overlap['only_second_pruned'],
        })

    # 汇总结果
    print("\n" + "=" * 100)
    print("实验结果汇总")
    print("=" * 100)
    print(f"原始损失: {loss_original:.4f}")
    print()
    print(f"{'剪枝率':<8} {'一阶损失':<10} {'一阶增加%':<12} {'二阶损失':<10} {'二阶增加%':<12} {'差异':<12} {'掩码重叠':<10}")
    print("-" * 95)

    for r in results:
        print(f"{r['target_ratio']*100:>5.0f}%   "
              f"{r['loss_first']:<10.4f} "
              f"{r['loss_inc_first_pct']:>+10.2f}%  "
              f"{r['loss_second']:<10.4f} "
              f"{r['loss_inc_second_pct']:>+10.2f}%  "
              f"{r['difference_pct']:>+10.2f}%  "
              f"{r['mask_overlap_jaccard']*100:>8.2f}%")

    # 保存结果
    pd.DataFrame(results).to_csv(output_dir / 'comparison.csv', index=False)

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
        'method': 'direct_percentile_threshold',
    }

    with open(output_dir / 'config.txt', 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    print(f"\n✓ 结果已保存到: {output_dir}")
    print("=" * 100)


if __name__ == '__main__':
    main()
