"""
三种重要性方法对比实验 - 使用完全相同的数据。

三种方法：
1. 一阶: d_i = |g_i * θ_i|
2. 一阶+二阶 (Adam): d_i = |g_i * θ_i| + α * |v_i * θ_i²|
3. HVP: d_i = |-g_i * θ_i + 0.5 * θ_i * (H * θ)_i|

关键：所有方法使用完全相同的数据批次计算梯度和评估损失。

运行命令:
    python experiments/scripts/compare_three_methods.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --prune_ratio 0.05 \
        --num_batches 10 \
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


def gpt2_loss_fn(model, batch):
    """GPT-2 损失函数。"""
    input_ids = batch['input_ids']
    labels = batch['labels']
    logits = model(input_ids)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    criterion = nn.CrossEntropyLoss()
    return criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def compute_gradients_and_momentum(model, cached_batches, device, num_batches):
    """计算梯度和 Adam 二阶矩。"""
    model.train()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    accumulated_gradients = defaultdict(lambda: 0)
    accumulated_exp_avg_sq = defaultdict(lambda: 0)

    print(f"  计算 {num_batches} 个批次的梯度和动量...")

    for step in tqdm(range(num_batches), desc="  梯度累积"):
        batch = cached_batches[step % len(cached_batches)]
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
        accumulated_gradients[name] = accumulated_gradients[name] / num_batches
        accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] / num_batches

    weights = {name: param.detach().cpu() for name, param in model.named_parameters()}
    return weights, dict(accumulated_gradients), dict(accumulated_exp_avg_sq)


def compute_hvp(model, loss_fn, data_batch, vector, device):
    """计算 Hessian-Vector Product。"""
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    batch_device = {k: v.to(device) for k, v in data_batch.items()}

    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        model.zero_grad()
        loss = loss_fn(model, batch_device)

        grads = torch.autograd.grad(
            loss, list(params.values()),
            create_graph=True, retain_graph=True,
        )

        grad_vector_product = torch.tensor(0.0, device=device)
        for g, name in zip(grads, params.keys()):
            if name in vector:
                v = vector[name].to(device)
                grad_vector_product = grad_vector_product + (g * v).sum()

        hvp_result = torch.autograd.grad(
            grad_vector_product, list(params.values()),
            retain_graph=False,
        )

    return {name: hvp.detach().cpu() for name, hvp in zip(params.keys(), hvp_result)}


def compute_hvp_batched(model, loss_fn, data_batches, vector, device, num_batches):
    """使用多个批次计算平均 HVP。"""
    hvp_sum = None
    actual_batches = min(num_batches, len(data_batches))

    for i in tqdm(range(actual_batches), desc="  HVP 计算"):
        hvp = compute_hvp(model, loss_fn, data_batches[i], vector, device)
        if hvp_sum is None:
            hvp_sum = {name: h.clone() for name, h in hvp.items()}
        else:
            for name in hvp_sum:
                hvp_sum[name] += hvp[name]

    return {name: h / actual_batches for name, h in hvp_sum.items()}


def compute_scores_first_order(weights, gradients):
    """一阶重要性: d_i = |g_i * θ_i|"""
    scores = {}
    for name, weight in weights.items():
        grad = gradients.get(name, torch.zeros_like(weight))
        scores[name] = torch.abs(grad * weight)
    return scores


def compute_scores_second_order(weights, gradients, exp_avg_sq, alpha=0.5):
    """一阶+二阶重要性: d_i = |g_i * θ_i| + α * |v_i * θ_i²|"""
    scores = {}
    for name, weight in weights.items():
        grad = gradients.get(name, torch.zeros_like(weight))
        v = exp_avg_sq.get(name, torch.zeros_like(weight))
        first_order = torch.abs(grad * weight)
        second_order = alpha * torch.abs(v * weight ** 2)
        scores[name] = first_order + second_order
    return scores


def compute_scores_hvp(model, loss_fn, data_batches, device, num_batches):
    """HVP 重要性: d_i = |-g_i * θ_i + 0.5 * θ_i * (H * θ)_i|"""
    model.train()
    model = model.to(device)

    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    weights = {name: p.data.clone().cpu() for name, p in params.items()}

    # 计算梯度
    batch_device = {k: v.to(device) for k, v in data_batches[0].items()}
    model.zero_grad()
    loss = loss_fn(model, batch_device)
    loss.backward()

    gradients = {
        name: p.grad.clone().cpu() if p.grad is not None else torch.zeros_like(p).cpu()
        for name, p in params.items()
    }

    # 计算 HVP
    hvp_result = compute_hvp_batched(model, loss_fn, data_batches, weights, device, num_batches)

    # 计算得分
    scores = {}
    for name in weights:
        theta = weights[name]
        grad = gradients.get(name, torch.zeros_like(theta))
        hvp = hvp_result.get(name, torch.zeros_like(theta))
        first_order = -grad * theta
        second_order = 0.5 * theta * hvp
        scores[name] = torch.abs(first_order + second_order)

    return scores


def get_prunable_scores(scores):
    """获取可剪枝层的得分（排除 Embedding 和 bias）。"""
    prunable = {}
    for name, score_tensor in scores.items():
        if 'weight' not in name:
            continue
        if 'wte.weight' in name or 'wpe.weight' in name:
            continue
        prunable[name] = score_tensor.clone()
    return prunable


def apply_global_pruning(model, scores, prune_ratio):
    """使用全局阈值进行剪枝。"""
    all_scores = torch.cat([s.flatten() for s in scores.values()])
    threshold = np.percentile(all_scores.numpy(), prune_ratio * 100)

    actual_pruned = 0
    total_params = 0

    for name, score_tensor in scores.items():
        mask = (score_tensor >= threshold).float()
        param = dict(model.named_parameters())[name]
        param.data.mul_(mask.to(param.device))
        actual_pruned += (mask == 0).sum().item()
        total_params += mask.numel()

    return model, actual_pruned / total_params, threshold


def evaluate_loss(model, cached_batches, device):
    """评估损失。"""
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description='三种重要性方法对比实验')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt')
    parser.add_argument('--prune_ratio', type=float, default=0.05)
    parser.add_argument('--num_batches', type=int, default=10,
                        help='用于计算梯度/HVP和评估的批次数（所有方法共用）')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output_dir', type=str, default='results/compare_three_methods')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("三种重要性方法对比实验")
    print("=" * 80)
    print(f"剪枝比例: {args.prune_ratio * 100:.0f}%")
    print(f"数据批次数: {args.num_batches} (所有方法共用)")
    print(f"输出目录: {output_dir}")
    print("=" * 80)

    # 1. 加载模型
    print("\n[1/6] 加载模型...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    print(f"✓ 检查点已加载 (训练步数: {checkpoint.get('step', 'unknown')})")

    # 2. 缓存数据批次（所有方法共用）
    print("\n[2/6] 缓存数据批次...")
    data_loader = get_wikitext103_dataloader(
        split='validation',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0
    )
    cached_batches = cache_batches(data_loader, args.num_batches)
    print(f"✓ 已缓存 {len(cached_batches)} 个批次（所有方法共用）")

    # 3. 评估原始模型
    print("\n[3/6] 评估原始模型...")
    model_original = get_gpt2_small(pretrained=False)
    model_original.load_state_dict(checkpoint['model_state_dict'])
    loss_original = evaluate_loss(model_original, cached_batches, device)
    print(f"✓ 原始损失: {loss_original:.4f}")

    # 4. 计算梯度和动量（一阶和二阶方法共用）
    print("\n[4/6] 计算梯度和动量...")
    model_for_grad = get_gpt2_small(pretrained=False)
    model_for_grad.load_state_dict(checkpoint['model_state_dict'])
    weights, gradients, exp_avg_sq = compute_gradients_and_momentum(
        model_for_grad, cached_batches, device, args.num_batches
    )
    print("✓ 梯度和动量已计算")

    # 5. 计算三种重要性得分
    print("\n[5/6] 计算三种重要性得分...")

    # 方法1: 一阶
    print("  [方法1] 一阶: d_i = |g_i * θ_i|")
    scores_first = compute_scores_first_order(weights, gradients)
    scores_first = get_prunable_scores(scores_first)

    # 方法2: 一阶+二阶
    print(f"  [方法2] 一阶+二阶: d_i = |g_i * θ_i| + {args.alpha} * |v_i * θ_i²|")
    scores_second = compute_scores_second_order(weights, gradients, exp_avg_sq, args.alpha)
    scores_second = get_prunable_scores(scores_second)

    # 方法3: HVP
    print("  [方法3] HVP: d_i = |-g_i * θ_i + 0.5 * θ_i * (H * θ)_i|")
    model_for_hvp = get_gpt2_small(pretrained=False)
    model_for_hvp.load_state_dict(checkpoint['model_state_dict'])
    scores_hvp = compute_scores_hvp(model_for_hvp, gpt2_loss_fn, cached_batches, device, args.num_batches)
    scores_hvp = get_prunable_scores(scores_hvp)

    print("✓ 三种重要性得分已计算")

    # 6. 剪枝并评估
    print(f"\n[6/6] 剪枝并评估 (剪枝比例: {args.prune_ratio * 100:.0f}%)...")

    results = []

    # 方法1: 一阶
    model_first = get_gpt2_small(pretrained=False)
    model_first.load_state_dict(checkpoint['model_state_dict'])
    model_first, actual_ratio_first, _ = apply_global_pruning(model_first, scores_first, args.prune_ratio)
    loss_first = evaluate_loss(model_first, cached_batches, device)
    loss_inc_first = (loss_first - loss_original) / loss_original * 100
    results.append({
        'method': '一阶',
        'formula': 'd_i = |g_i * θ_i|',
        'actual_ratio': actual_ratio_first,
        'loss_pruned': loss_first,
        'loss_increase_pct': loss_inc_first,
    })
    print(f"  一阶:      实际剪枝率={actual_ratio_first*100:.2f}%, 损失={loss_first:.4f}, 增加={loss_inc_first:+.2f}%")

    # 方法2: 一阶+二阶
    model_second = get_gpt2_small(pretrained=False)
    model_second.load_state_dict(checkpoint['model_state_dict'])
    model_second, actual_ratio_second, _ = apply_global_pruning(model_second, scores_second, args.prune_ratio)
    loss_second = evaluate_loss(model_second, cached_batches, device)
    loss_inc_second = (loss_second - loss_original) / loss_original * 100
    results.append({
        'method': '一阶+二阶',
        'formula': f'd_i = |g_i * θ_i| + {args.alpha} * |v_i * θ_i²|',
        'actual_ratio': actual_ratio_second,
        'loss_pruned': loss_second,
        'loss_increase_pct': loss_inc_second,
    })
    print(f"  一阶+二阶: 实际剪枝率={actual_ratio_second*100:.2f}%, 损失={loss_second:.4f}, 增加={loss_inc_second:+.2f}%")

    # 方法3: HVP
    model_hvp = get_gpt2_small(pretrained=False)
    model_hvp.load_state_dict(checkpoint['model_state_dict'])
    model_hvp, actual_ratio_hvp, _ = apply_global_pruning(model_hvp, scores_hvp, args.prune_ratio)
    loss_hvp = evaluate_loss(model_hvp, cached_batches, device)
    loss_inc_hvp = (loss_hvp - loss_original) / loss_original * 100
    results.append({
        'method': 'HVP',
        'formula': 'd_i = |-g_i * θ_i + 0.5 * θ_i * (H * θ)_i|',
        'actual_ratio': actual_ratio_hvp,
        'loss_pruned': loss_hvp,
        'loss_increase_pct': loss_inc_hvp,
    })
    print(f"  HVP:       实际剪枝率={actual_ratio_hvp*100:.2f}%, 损失={loss_hvp:.4f}, 增加={loss_inc_hvp:+.2f}%")

    # 汇总结果
    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)
    print(f"原始损失: {loss_original:.4f}")
    print(f"剪枝比例: {args.prune_ratio * 100:.0f}%")
    print(f"数据批次: {args.num_batches} 个（所有方法共用同一批数据）")
    print()
    print(f"{'方法':<12} {'公式':<45} {'实际剪枝率':<12} {'剪枝损失':<10} {'损失增加%':<10}")
    print("-" * 95)

    for r in results:
        print(f"{r['method']:<12} {r['formula']:<45} {r['actual_ratio']*100:>8.2f}%    "
              f"{r['loss_pruned']:<10.4f} {r['loss_increase_pct']:>+8.2f}%")

    # 保存结果
    for r in results:
        r['loss_original'] = loss_original
        r['target_ratio'] = args.prune_ratio

    pd.DataFrame(results).to_csv(output_dir / 'results.csv', index=False)

    config = {
        'checkpoint': args.checkpoint,
        'prune_ratio': args.prune_ratio,
        'num_batches': args.num_batches,
        'batch_size': args.batch_size,
        'seq_length': args.seq_length,
        'alpha': args.alpha,
        'device': device,
        'loss_original': loss_original,
    }

    with open(output_dir / 'config.txt', 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    print(f"\n✓ 结果已保存到: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
