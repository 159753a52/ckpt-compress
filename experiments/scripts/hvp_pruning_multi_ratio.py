"""
基于 HVP (Hessian-Vector Product) 的参数重要性剪枝实验。

HVP 重要性公式: s_i = -g_i * θ_i + 0.5 * θ_i * (H * θ)_i

其中：
- g_i: 参数的梯度
- θ_i: 参数值
- (H * θ)_i: Hessian-Vector Product 的第 i 个元素

相比 Adam 二阶矩近似，HVP 方法：
- 考虑了 Hessian 的非对角元素（参数间相互作用）
- 计算更准确，但速度较慢
- 需要两次反向传播

运行命令:
    python experiments/scripts/hvp_pruning_multi_ratio.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --prune_ratios 0.10,0.15,0.20,0.25,0.30 \
        --hvp_batches 5 \
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
    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def compute_hvp(model, loss_fn, data_batch, vector, device):
    """
    计算 Hessian-Vector Product: H * v

    使用两次反向传播计算 HVP，不需要显式构建 Hessian 矩阵。
    原理：H * v = ∂/∂θ (∇L · v)
    """
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}

    # 将数据移到设备
    batch_device = {k: v.to(device) for k, v in data_batch.items()}

    # 禁用 scaled_dot_product_attention 以支持二阶导数
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        # 第一次前向传播和反向传播，计算梯度
        model.zero_grad()
        loss = loss_fn(model, batch_device)

        # 计算一阶梯度，保留计算图
        grads = torch.autograd.grad(
            loss,
            list(params.values()),
            create_graph=True,
            retain_graph=True,
        )

        # 计算 grad · vector 的标量积
        grad_vector_product = torch.tensor(0.0, device=device)
        for g, name in zip(grads, params.keys()):
            if name in vector:
                v = vector[name].to(device)
                grad_vector_product = grad_vector_product + (g * v).sum()

        # 第二次反向传播，计算 HVP
        hvp_result = torch.autograd.grad(
            grad_vector_product,
            list(params.values()),
            retain_graph=False,
        )

    # 转换为字典格式
    hvp_dict = {
        name: hvp.detach().cpu()
        for name, hvp in zip(params.keys(), hvp_result)
    }

    return hvp_dict


def compute_hvp_batched(model, loss_fn, data_batches, vector, device, num_batches=1):
    """使用多个批次计算平均 HVP。"""
    hvp_sum = None
    actual_batches = min(num_batches, len(data_batches))

    for i in tqdm(range(actual_batches), desc="计算 HVP"):
        batch = data_batches[i]
        hvp = compute_hvp(model, loss_fn, batch, vector, device)

        if hvp_sum is None:
            hvp_sum = {name: h.clone() for name, h in hvp.items()}
        else:
            for name in hvp_sum:
                hvp_sum[name] += hvp[name]

    if hvp_sum is None:
        return {}

    hvp_avg = {name: h / actual_batches for name, h in hvp_sum.items()}
    return hvp_avg


def compute_importance_scores_hvp(model, loss_fn, data_batches, device, num_batches=1):
    """
    使用 HVP 计算参数重要性得分。

    公式: s_i = -g_i * θ_i + 0.5 * θ_i * (H * θ)_i
    """
    model.train()
    model = model.to(device)

    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    weights = {name: p.data.clone().cpu() for name, p in params.items()}

    # 计算梯度（用于一阶项）
    print("计算梯度...")
    batch_device = {k: v.to(device) for k, v in data_batches[0].items()}
    model.zero_grad()
    loss = loss_fn(model, batch_device)
    loss.backward()

    gradients = {
        name: p.grad.clone().cpu() if p.grad is not None else torch.zeros_like(p).cpu()
        for name, p in params.items()
    }

    # 计算 HVP: H * θ
    print(f"计算 HVP ({num_batches} 批次)...")
    hvp_result = compute_hvp_batched(model, loss_fn, data_batches, weights, device, num_batches)

    # 计算重要性得分: s_i = -g_i * θ_i + 0.5 * θ_i * (H * θ)_i
    scores = {}
    for name in weights:
        theta = weights[name]
        grad = gradients.get(name, torch.zeros_like(theta))
        hvp = hvp_result.get(name, torch.zeros_like(theta))

        first_order = -grad * theta
        second_order = 0.5 * theta * hvp

        scores[name] = first_order + second_order

    return scores


def compute_importance_scores_hvp_abs(model, loss_fn, data_batches, device, num_batches=1):
    """
    使用 HVP 计算参数重要性得分（绝对值版本）。

    公式: d_i = |-g_i * θ_i + 0.5 * θ_i * (H * θ)_i|
    """
    scores = compute_importance_scores_hvp(model, loss_fn, data_batches, device, num_batches)

    # 取绝对值
    scores_abs = {name: torch.abs(s) for name, s in scores.items()}
    return scores_abs


def get_prunable_scores(scores):
    """获取可剪枝层的得分（排除 Embedding 和 bias）。"""
    prunable = {}
    for name, score_tensor in scores.items():
        if 'weight' not in name:
            continue
        # 排除 Embedding 层
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

    actual_ratio = actual_pruned / total_params
    return model, actual_ratio, threshold


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


def main():
    import argparse

    parser = argparse.ArgumentParser(description='HVP 重要性剪枝实验')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt')
    parser.add_argument('--prune_ratios', type=str, default='0.10,0.15,0.20,0.25,0.30')
    parser.add_argument('--hvp_batches', type=int, default=5,
                        help='用于计算 HVP 的批次数')
    parser.add_argument('--eval_batches', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output_dir', type=str, default='results/hvp_pruning_multi')
    parser.add_argument('--use_abs', action='store_true', default=True,
                        help='使用绝对值版本的重要性得分')

    args = parser.parse_args()

    prune_ratios = [float(x.strip()) for x in args.prune_ratios.split(',')]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 90)
    print("HVP (Hessian-Vector Product) 重要性剪枝实验")
    print("=" * 90)
    print(f"重要性公式: s_i = -g_i * θ_i + 0.5 * θ_i * (H * θ)_i")
    print(f"使用绝对值: {args.use_abs}")
    print(f"剪枝比例: {[f'{r*100:.0f}%' for r in prune_ratios]}")
    print(f"HVP 批次数: {args.hvp_batches}")
    print(f"评估批次数: {args.eval_batches}")
    print(f"输出目录: {output_dir}")
    print("=" * 90)

    # 1. 加载模型
    print("\n[1/5] 加载模型...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model = get_gpt2_small(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 模型已加载 (训练步数: {checkpoint.get('step', 'unknown')})")

    # 2. 加载数据
    print("\n[2/5] 加载数据...")
    data_loader = get_wikitext103_dataloader(
        split='train',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0
    )
    cached_train_batches = cache_batches(data_loader, max(args.hvp_batches + 5, 20))
    print(f"  ✓ 已缓存 {len(cached_train_batches)} 个训练批次")

    data_loader_eval = get_wikitext103_dataloader(
        split='validation',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0
    )
    cached_eval_batches = cache_batches(data_loader_eval, args.eval_batches)
    print(f"  ✓ 已缓存 {len(cached_eval_batches)} 个评估批次")

    # 3. 计算 HVP 重要性得分
    print("\n[3/5] 计算 HVP 重要性得分...")
    model = get_gpt2_small(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if args.use_abs:
        scores = compute_importance_scores_hvp_abs(
            model, gpt2_loss_fn, cached_train_batches, device, args.hvp_batches
        )
    else:
        scores = compute_importance_scores_hvp(
            model, gpt2_loss_fn, cached_train_batches, device, args.hvp_batches
        )

    # 过滤可剪枝层
    scores = get_prunable_scores(scores)
    print(f"✓ 已计算 {len(scores)} 层的重要性得分（排除 Embedding）")

    # 分析得分
    all_scores = torch.cat([s.flatten() for s in scores.values()]).numpy()
    print(f"  得分统计: mean={np.mean(all_scores):.6e}, std={np.std(all_scores):.6e}")
    print(f"            min={np.min(all_scores):.6e}, max={np.max(all_scores):.6e}")

    # 4. 评估原始模型
    print("\n[4/5] 评估原始模型...")
    model_original = get_gpt2_small(pretrained=False)
    model_original.load_state_dict(checkpoint['model_state_dict'])
    loss_original = evaluate_loss_with_cached_batches(model_original, cached_eval_batches, device)
    print(f"✓ 原始损失: {loss_original:.4f}")

    # 5. 对每个剪枝比例运行实验
    print("\n[5/5] 运行剪枝实验...")
    results = []

    for prune_ratio in prune_ratios:
        print(f"\n--- 剪枝比例: {prune_ratio*100:.0f}% ---")

        model_pruned = get_gpt2_small(pretrained=False)
        model_pruned.load_state_dict(checkpoint['model_state_dict'])

        model_pruned, actual_ratio, threshold = apply_global_pruning(
            model_pruned, scores, prune_ratio
        )

        loss_pruned = evaluate_loss_with_cached_batches(model_pruned, cached_eval_batches, device)
        loss_increase = loss_pruned - loss_original
        loss_increase_pct = (loss_increase / loss_original) * 100

        print(f"  实际剪枝率: {actual_ratio*100:.2f}%")
        print(f"  全局阈值: {threshold:.6e}")
        print(f"  剪枝后损失: {loss_pruned:.4f}")
        print(f"  损失增加: {loss_increase:.4f} ({loss_increase_pct:+.2f}%)")

        results.append({
            'target_ratio': prune_ratio,
            'actual_ratio': actual_ratio,
            'threshold': threshold,
            'loss_original': loss_original,
            'loss_pruned': loss_pruned,
            'loss_increase': loss_increase,
            'loss_increase_pct': loss_increase_pct,
        })

    # 汇总结果
    print("\n" + "=" * 80)
    print("实验结果汇总 (HVP 重要性)")
    print("=" * 80)
    print(f"原始损失: {loss_original:.4f}")
    print()
    print(f"{'剪枝率':<10} {'实际率':<10} {'剪枝损失':<12} {'损失增加':<12} {'增加%':<10}")
    print("-" * 60)

    for r in results:
        print(f"{r['target_ratio']*100:>6.0f}%    "
              f"{r['actual_ratio']*100:>6.2f}%    "
              f"{r['loss_pruned']:<12.4f} "
              f"{r['loss_increase']:<12.4f} "
              f"{r['loss_increase_pct']:>+8.2f}%")

    # 保存结果
    pd.DataFrame(results).to_csv(output_dir / 'results.csv', index=False)

    config = {
        'checkpoint': args.checkpoint,
        'prune_ratios': prune_ratios,
        'hvp_batches': args.hvp_batches,
        'eval_batches': args.eval_batches,
        'batch_size': args.batch_size,
        'seq_length': args.seq_length,
        'device': device,
        'use_abs': args.use_abs,
        'loss_original': loss_original,
        'importance_formula': 's_i = -g_i * θ_i + 0.5 * θ_i * (H * θ)_i',
    }

    with open(output_dir / 'config.txt', 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    print(f"\n✓ 结果已保存到: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
