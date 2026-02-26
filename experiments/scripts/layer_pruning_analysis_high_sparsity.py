"""
逐层剪枝对损失的影响分析 - 高稀疏度测试

测试Block 0的Attention、MLP、LayerNorm层在高稀疏度（40%-90%）下的表现。

运行命令:
    python experiments/scripts/layer_pruning_analysis_high_sparsity.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --num_steps 100 \
        --batch_size 4 \
        --seq_length 512 \
        --num_eval_batches 10 \
        --device cuda \
        --output_dir results/layer_pruning_analysis_high_sparsity
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
    """累积多个步骤的梯度并计算重要性得分。"""
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

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

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

    print("计算重要性得分...")
    scores = compute_importance_scores_first_order(
        weights=weights,
        gradients=dict(accumulated_gradients),
        exp_avg_sq=dict(accumulated_exp_avg_sq),
        alpha=0.5
    )

    return scores


def select_layer_params(layer_group, all_param_names):
    """选择指定层次的参数名称。"""
    if layer_group == 'block0_attn':
        return [n for n in all_param_names if 'transformer.h.0.attn' in n]
    elif layer_group == 'block0_mlp':
        return [n for n in all_param_names if 'transformer.h.0.mlp' in n]
    elif layer_group == 'block0_ln':
        return [n for n in all_param_names if 'transformer.h.0.ln' in n]
    else:
        raise ValueError(f"Unknown layer group: {layer_group}")


def evaluate_loss(model, eval_batches, device):
    """评估模型在多个批次上的平均损失。"""
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
    """剪枝指定层次并评估损失。"""
    layer_scores = {name: scores[name] for name in layer_params}
    flat_scores = torch.cat([s.flatten() for s in layer_scores.values()])

    # 计算阈值（对大张量使用采样）
    if flat_scores.numel() > 10_000_000:
        sample_size = 10_000_000
        indices = torch.randperm(flat_scores.numel())[:sample_size]
        sampled_scores = flat_scores[indices]
        threshold = torch.quantile(sampled_scores, sparsity)
    else:
        threshold = torch.quantile(flat_scores, sparsity)

    # 创建剪枝掩码
    masks = {}
    for name in layer_params:
        mask = (layer_scores[name] >= threshold).float()
        masks[name] = mask

    # 保存原始权重
    original_weights = {}
    param_dict = dict(model.named_parameters())
    for name in layer_params:
        param = param_dict[name]
        original_weights[name] = param.data.clone()

    # 应用剪枝
    for name in layer_params:
        param = param_dict[name]
        param.data.mul_(masks[name].to(device))

    # 评估损失
    avg_loss = evaluate_loss(model, eval_batches, device)

    # 恢复权重
    for name in layer_params:
        param = param_dict[name]
        param.data.copy_(original_weights[name])

    # 统计剪枝信息
    pruned_params = (flat_scores < threshold).sum().item()
    total_params = flat_scores.numel()

    return {
        'loss': avg_loss,
        'pruned_params': pruned_params,
        'total_params': total_params,
    }


def save_results(results, output_dir):
    """保存结果为 CSV 和图表。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = output_dir / 'pruning_results_high_sparsity.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV 已保存: {csv_path}")

    # 图表1: Loss vs Sparsity
    fig, ax = plt.subplots(figsize=(12, 7))

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

    ax.set_xlabel('Sparsity (%)', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Loss vs Sparsity (High Sparsity Range: 40%-90%)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plot1_path = output_dir / 'loss_vs_sparsity_high.png'
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表 1 已保存: {plot1_path}")
    plt.close()

    # 图表2: Loss Increase vs Sparsity
    fig, ax = plt.subplots(figsize=(12, 7))

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

    ax.set_xlabel('Sparsity (%)', fontsize=13)
    ax.set_ylabel('Loss Increase (Δ Loss)', fontsize=13)
    ax.set_title('Loss Increase vs Sparsity (High Sparsity Range: 40%-90%)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plot2_path = output_dir / 'loss_increase_vs_sparsity_high.png'
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表 2 已保存: {plot2_path}")
    plt.close()


def run_pruning_experiment(checkpoint_path, output_dir, device='cuda',
                          num_steps=100, batch_size=4, seq_length=512,
                          num_eval_batches=10):
    """运行高稀疏度剪枝实验。"""
    print("=" * 80)
    print("逐层剪枝对损失的影响分析 - 高稀疏度测试（40%-90%）")
    print("=" * 80)
    print(f"检查点: {checkpoint_path}")
    print(f"梯度累积步数: {num_steps}")
    print(f"评估批次数: {num_eval_batches}")
    print(f"重要性度量: |g_i * θ_i|")
    print(f"测试层次: Block 0 - Attention, MLP, LayerNorm")
    print("=" * 80)

    # 加载模型和数据
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

    # 准备评估批次
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

    # 计算基线损失
    print("\n计算基线损失...")
    baseline_loss = evaluate_loss(model, eval_batches, device)
    print(f"✓ 基线损失: {baseline_loss:.4f}")

    # 累积梯度并计算重要性
    print("\n累积梯度并计算重要性得分...")
    scores = collect_gradients_and_compute_importance(
        model, data_loader, device, num_steps=num_steps
    )
    print(f"✓ 已计算重要性得分")

    # 定义实验组（不包括Embedding）
    layer_groups = {
        'block0_attn': 'Block 0 - Attention',
        'block0_mlp': 'Block 0 - MLP',
        'block0_ln': 'Block 0 - LayerNorm',
    }

    # 高稀疏度：40%-90%，每次增加10%
    sparsities = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    # 运行实验
    print("\n" + "=" * 80)
    print("开始高稀疏度剪枝实验")
    print("=" * 80)

    results = []
    all_param_names = list(scores.keys())

    for layer_group, layer_name in layer_groups.items():
        print(f"\n实验组: {layer_name}")

        layer_params = select_layer_params(layer_group, all_param_names)
        print(f"  参数数量: {len(layer_params)}")

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

    # 保存结果
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
    parser = argparse.ArgumentParser(description='逐层剪枝对损失的影响分析 - 高稀疏度测试')
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
                        default='results/layer_pruning_analysis_high_sparsity',
                        help='输出目录')

    args = parser.parse_args()

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
