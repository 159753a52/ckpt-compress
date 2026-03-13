"""
逐层剪枝对损失的影响分析

使用一阶梯度绝对值 |g·θ| 作为重要性度量，分析不同层次的参数剪枝对模型损失的影响。

运行命令:
    python experiments/scripts/layer_pruning_analysis.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --num_steps 100 \
        --batch_size 4 \
        --seq_length 512 \
        --num_eval_batches 10 \
        --device cuda \
        --output_dir results/layer_pruning_analysis

实验设计:
    - 层次分组: Embedding, Block 0 (Attention, MLP, LayerNorm)
    - 稀疏度: 5%, 10%, 15%, 20%, 25%, 30%
    - 重要性度量: |g_i * θ_i|
    - 评估: 多批次平均损失
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm
from collections import defaultdict
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader
from src.ckpt_compress.methods.adam_prune.importance import compute_importance_scores_first_order


def collect_gradients_and_compute_importance(model, data_loader, device, num_steps=100):
    """
    累积多个步骤的梯度并计算重要性得分。

    参数:
        model: GPT-2 模型
        data_loader: 数据加载器
        device: 设备 (cuda/cpu)
        num_steps: 累积的训练步数

    返回:
        scores: 重要性得分字典 {name: tensor}
    """
    model.train()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    accumulated_gradients = defaultdict(lambda: 0)
    accumulated_exp_avg_sq = defaultdict(lambda: 0)

    print(f"累积 {num_steps} 步的梯度...")

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

        # 计算损失（shift logits 和 labels）
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        loss.backward()

        # 累积梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                accumulated_gradients[name] = accumulated_gradients[name] + param.grad.detach().cpu()

        optimizer.step()

        # 累积 exp_avg_sq
        for name, param in model.named_parameters():
            if name in optimizer.state[param]:
                exp_avg_sq = optimizer.state[param]['exp_avg_sq']
                accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] + exp_avg_sq.detach().cpu()

    # 取平均
    for name in accumulated_gradients:
        accumulated_gradients[name] = accumulated_gradients[name] / num_steps
        accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] / num_steps

    # 收集权重
    weights = {name: param.detach().cpu() for name, param in model.named_parameters()}

    # 计算重要性得分（一阶项）
    print("计算重要性得分...")
    scores = compute_importance_scores_first_order(
        weights=weights,
        gradients=dict(accumulated_gradients),
        exp_avg_sq=dict(accumulated_exp_avg_sq),
        alpha=0.5  # 未使用
    )

    return scores


def select_layer_params(layer_group, all_param_names):
    """
    选择指定层次的参数名称。

    参数:
        layer_group: 层次组名称
        all_param_names: 所有参数名称列表

    返回:
        选中的参数名称列表
    """
    if layer_group == 'embedding':
        return [n for n in all_param_names if 'wte.weight' in n or 'wpe.weight' in n]

    elif layer_group == 'block0_attn':
        return [n for n in all_param_names if 'transformer.h.0.attn' in n]

    elif layer_group == 'block0_mlp':
        return [n for n in all_param_names if 'transformer.h.0.mlp' in n]

    elif layer_group == 'block0_ln':
        return [n for n in all_param_names if 'transformer.h.0.ln' in n]

    else:
        raise ValueError(f"Unknown layer group: {layer_group}")


def evaluate_loss(model, eval_batches, device):
    """
    评估模型在多个批次上的平均损失。

    参数:
        model: GPT-2 模型
        eval_batches: 评估批次列表
        device: 设备

    返回:
        平均损失
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch in eval_batches:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(eval_batches)
    return avg_loss


def prune_and_evaluate(model, layer_params, scores, sparsity, eval_batches, device):
    """
    剪枝指定层次并评估损失。

    参数:
        model: GPT-2 模型
        layer_params: 要剪枝的参数名称列表
        scores: 重要性得分字典
        sparsity: 稀疏度 (0-1)
        eval_batches: 评估批次列表
        device: 设备

    返回:
        结果字典 {loss, pruned_params, total_params}
    """
    # 1. 获取该层次的重要性得分
    layer_scores = {name: scores[name] for name in layer_params}

    # 2. 展平得分
    flat_scores = torch.cat([s.flatten() for s in layer_scores.values()])

    # 3. 计算阈值（对于大张量使用采样）
    if flat_scores.numel() > 10_000_000:
        # 对于超过1000万参数的层，使用采样计算阈值
        sample_size = 10_000_000
        indices = torch.randperm(flat_scores.numel())[:sample_size]
        sampled_scores = flat_scores[indices]
        threshold = torch.quantile(sampled_scores, sparsity)
    else:
        threshold = torch.quantile(flat_scores, sparsity)

    # 4. 创建剪枝掩码
    masks = {}
    for name in layer_params:
        mask = (layer_scores[name] >= threshold).float()
        masks[name] = mask

    # 5. 保存原始权重
    original_weights = {}
    param_dict = dict(model.named_parameters())
    for name in layer_params:
        param = param_dict[name]
        original_weights[name] = param.data.clone()

    # 6. 应用剪枝
    for name in layer_params:
        param = param_dict[name]
        param.data.mul_(masks[name].to(device))

    # 7. 评估损失（多批次平均）
    avg_loss = evaluate_loss(model, eval_batches, device)

    # 8. 恢复权重
    for name in layer_params:
        param = param_dict[name]
        param.data.copy_(original_weights[name])

    # 9. 统计剪枝信息
    pruned_params = (flat_scores < threshold).sum().item()
    total_params = flat_scores.numel()

    return {
        'loss': avg_loss,
        'pruned_params': pruned_params,
        'total_params': total_params,
    }


def save_results(results, output_dir):
    """
    保存结果为 CSV 和图表。

    参数:
        results: 结果列表
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存 CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / 'pruning_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV 已保存: {csv_path}")

    # 2. 绘制图表 1: Loss vs Sparsity
    fig, ax = plt.subplots(figsize=(10, 6))

    for layer_group in df['layer'].unique():
        layer_data = df[df['layer'] == layer_group]
        ax.plot(
            layer_data['sparsity'] * 100,
            layer_data['loss'],
            marker='o',
            label=layer_data['layer_name'].iloc[0],
            linewidth=2,
            markersize=8
        )

    ax.axhline(df['baseline_loss'].iloc[0], color='black',
               linestyle='--', label='Baseline', linewidth=2)

    ax.set_xlabel('Sparsity (%)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss vs Sparsity for Different Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plot1_path = output_dir / 'loss_vs_sparsity.png'
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表 1 已保存: {plot1_path}")
    plt.close()

    # 3. 绘制图表 2: Loss Increase vs Sparsity
    fig, ax = plt.subplots(figsize=(10, 6))

    for layer_group in df['layer'].unique():
        layer_data = df[df['layer'] == layer_group]
        ax.plot(
            layer_data['sparsity'] * 100,
            layer_data['loss_increase'],
            marker='o',
            label=layer_data['layer_name'].iloc[0],
            linewidth=2,
            markersize=8
        )

    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel('Sparsity (%)', fontsize=12)
    ax.set_ylabel('Loss Increase (Δ Loss)', fontsize=12)
    ax.set_title('Loss Increase vs Sparsity for Different Layers',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plot2_path = output_dir / 'loss_increase_vs_sparsity.png'
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表 2 已保存: {plot2_path}")
    plt.close()

    # 4. 绘制图表 3: 参数数量统计
    fig, ax = plt.subplots(figsize=(10, 6))

    layer_groups = df['layer'].unique()
    layer_names = [df[df['layer'] == lg]['layer_name'].iloc[0] for lg in layer_groups]
    total_params = [df[df['layer'] == lg]['total_params'].iloc[0] for lg in layer_groups]

    bars = ax.bar(range(len(layer_names)), total_params, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylabel('Number of Parameters', fontsize=12)
    ax.set_title('Parameter Count by Layer', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars, total_params)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=9)

    plot3_path = output_dir / 'parameter_counts.png'
    plt.tight_layout()
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表 3 已保存: {plot3_path}")
    plt.close()


def run_pruning_experiment(checkpoint_path, output_dir, device='cuda',
                          num_steps=100, batch_size=4, seq_length=512,
                          num_eval_batches=10):
    """
    运行剪枝实验。

    参数:
        checkpoint_path: 检查点路径
        output_dir: 输出目录
        device: 设备
        num_steps: 梯度累积步数
        batch_size: 批次大小
        seq_length: 序列长度
        num_eval_batches: 评估批次数

    返回:
        results: 结果列表
    """
    print("=" * 80)
    print("逐层剪枝对损失的影响分析")
    print("=" * 80)
    print(f"检查点: {checkpoint_path}")
    print(f"梯度累积步数: {num_steps}")
    print(f"评估批次数: {num_eval_batches}")
    print(f"重要性度量: |g_i * θ_i|")
    print("=" * 80)

    # 1. 加载模型和数据
    print("\n加载模型...")
    model = get_gpt2_small(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✓ 模型已加载 (步数: {checkpoint.get('step', 'unknown')})")

    print("\n加载数据...")
    data_loader = get_wikitext103_dataloader(
        split='train',
        batch_size=batch_size,
        seq_length=seq_length,
        num_workers=0
    )
    print(f"✓ 数据已加载")

    # 2. 准备评估批次
    print(f"\n准备 {num_eval_batches} 个评估批次...")
    eval_batches = []
    eval_iter = iter(data_loader)
    for i in range(num_eval_batches):
        try:
            batch = next(eval_iter)
            eval_batches.append(batch)
        except StopIteration:
            print(f"警告: 只能获取 {i} 个批次")
            break
    print(f"✓ 已准备 {len(eval_batches)} 个评估批次")

    # 3. 计算基线损失（未剪枝）
    print("\n计算基线损失...")
    baseline_loss = evaluate_loss(model, eval_batches, device)
    print(f"✓ 基线损失: {baseline_loss:.4f}")

    # 4. 累积梯度并计算重要性
    print("\n累积梯度并计算重要性得分...")
    scores = collect_gradients_and_compute_importance(
        model, data_loader, device, num_steps=num_steps
    )
    print(f"✓ 已计算重要性得分")

    # 5. 定义实验组
    layer_groups = {
        'embedding': 'Embedding Layer',
        'block0_attn': 'Block 0 - Attention',
        'block0_mlp': 'Block 0 - MLP',
        'block0_ln': 'Block 0 - LayerNorm',
    }

    sparsities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    # 6. 运行实验
    print("\n" + "=" * 80)
    print("开始剪枝实验")
    print("=" * 80)

    results = []
    all_param_names = list(scores.keys())

    for layer_group, layer_name in layer_groups.items():
        print(f"\n实验组: {layer_name}")

        layer_params = select_layer_params(layer_group, all_param_names)
        print(f"  参数数量: {len(layer_params)}")

        # 打印参数名称
        for param_name in layer_params:
            param_size = scores[param_name].numel()
            print(f"    - {param_name}: {param_size:,} 参数")

        for sparsity in sparsities:
            result = prune_and_evaluate(
                model, layer_params, scores, sparsity, eval_batches, device
            )

            result['layer'] = layer_group
            result['layer_name'] = layer_name
            result['sparsity'] = sparsity
            result['baseline_loss'] = baseline_loss
            result['loss_increase'] = result['loss'] - baseline_loss

            results.append(result)

            print(f"    稀疏度 {sparsity*100:5.1f}%: "
                  f"loss={result['loss']:.4f}, "
                  f"Δloss={result['loss_increase']:+.4f}, "
                  f"剪枝={result['pruned_params']:,}/{result['total_params']:,}")

    # 7. 保存结果
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)
    save_results(results, output_dir)

    print("\n" + "=" * 80)
    print("实验完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description='逐层剪枝对损失的影响分析')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='检查点路径')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='梯度累积步数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='序列长度')
    parser.add_argument('--num_eval_batches', type=int, default=10,
                        help='评估批次数')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备')
    parser.add_argument('--output_dir', type=str,
                        default='results/layer_pruning_analysis',
                        help='输出目录')

    args = parser.parse_args()

    # 检查设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = 'cpu'

    results = run_pruning_experiment(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=device,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_eval_batches=args.num_eval_batches
    )


if __name__ == '__main__':
    main()
