"""
细粒度剪枝分析 - 只关注 weight 参数

分析 GPT-2 Small 所有 12 个 blocks 中每个 weight 参数的无损剪枝比例。

细粒度层次划分:
    Attention:
        - attn_qkv: c_attn.weight [768, 2304] (QKV 投影矩阵)
        - attn_proj: c_proj.weight [768, 768] (输出投影矩阵)
    MLP:
        - mlp_fc: c_fc.weight [768, 3072] (第一层/扩展层)
        - mlp_proj: c_proj.weight [3072, 768] (第二层/投影层)
    LayerNorm:
        - ln_1: ln_1.weight [768] (Attention 前)
        - ln_2: ln_2.weight [768] (MLP 前)

运行命令:
    python experiments/scripts/fine_grained_pruning_analysis.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --num_steps 100 \
        --batch_size 4 \
        --seq_length 512 \
        --num_eval_batches 10 \
        --device cuda \
        --output_dir results/fine_grained_pruning_analysis

实验设计:
    - 测试范围: 所有 12 个 Transformer Blocks + Embedding
    - 参数组: attn_qkv, attn_proj, mlp_fc, mlp_proj, ln_1, ln_2 (每个 block)
    - 稀疏度: 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%, 55%, 60%, 65%, 70%, 75%, 80%, 85%, 90%
    - 重要性度量: |g_i * θ_i|
    - 无损标准: 损失增量 <= 0.001
    - 只测试 weight 参数（排除 bias）
"""

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from tqdm import tqdm
import argparse
from datetime import datetime
import gc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader
from src.ckpt_compress.methods.adam_prune.importance import compute_importance_scores_first_order


def collect_gradients_and_compute_importance(model, data_loader, device, num_steps=100):
    """累积梯度并计算重要性得分（内存优化版本）"""
    model.train()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # 初始化累积字典（避免重复创建张量）
    accumulated_gradients = {}
    accumulated_exp_avg_sq = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            accumulated_gradients[name] = torch.zeros_like(param, device='cpu')
            accumulated_exp_avg_sq[name] = torch.zeros_like(param, device='cpu')

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

        # 累积梯度（原地操作）
        for name, param in model.named_parameters():
            if param.grad is not None:
                accumulated_gradients[name].add_(param.grad.detach().cpu())

        optimizer.step()

        # 累积 exp_avg_sq（原地操作）
        for name, param in model.named_parameters():
            if name in optimizer.state[param]:
                exp_avg_sq = optimizer.state[param]['exp_avg_sq']
                accumulated_exp_avg_sq[name].add_(exp_avg_sq.detach().cpu())

        # 释放中间变量
        del input_ids, labels, logits, shift_logits, shift_labels, loss

        # 定期清理GPU缓存
        if device == 'cuda' and step % 10 == 0:
            torch.cuda.empty_cache()

    # 取平均（原地操作）
    for name in accumulated_gradients:
        accumulated_gradients[name].div_(num_steps)
        accumulated_exp_avg_sq[name].div_(num_steps)

    # 收集权重
    weights = {name: param.detach().cpu() for name, param in model.named_parameters()}

    # 计算重要性得分
    print("计算重要性得分...")
    scores = compute_importance_scores_first_order(
        weights=weights,
        gradients=accumulated_gradients,
        exp_avg_sq=accumulated_exp_avg_sq,
        alpha=0.5
    )

    # 清理内存
    del accumulated_gradients, accumulated_exp_avg_sq, weights
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return scores


def select_layer_params(layer_group, block_id, all_param_names):
    """
    选择指定层次的参数名称（只选择 weight 参数）

    参数:
        layer_group: 层次类型
            - 'embedding': Token + Position Embedding
            - 'attn_qkv': Attention QKV 投影矩阵
            - 'attn_proj': Attention 输出投影矩阵
            - 'mlp_fc': MLP 第一层（扩展层）
            - 'mlp_proj': MLP 第二层（投影层）
            - 'ln_1': LayerNorm 1 (Attention 前)
            - 'ln_2': LayerNorm 2 (MLP 前)
        block_id: Block 编号 (0-11)，Embedding 层为 -1
        all_param_names: 所有参数名称列表

    返回:
        匹配的参数名称列表
    """
    if layer_group == 'embedding':
        # 只选择 embedding 的 weight 参数
        return [n for n in all_param_names if ('wte.weight' in n or 'wpe.weight' in n)]

    elif layer_group == 'attn_qkv':
        # Attention QKV 投影矩阵
        return [n for n in all_param_names
                if f'transformer.h.{block_id}.attn.c_attn.weight' in n]

    elif layer_group == 'attn_proj':
        # Attention 输出投影矩阵
        return [n for n in all_param_names
                if f'transformer.h.{block_id}.attn.c_proj.weight' in n]

    elif layer_group == 'mlp_fc':
        # MLP 第一层（扩展层）
        return [n for n in all_param_names
                if f'transformer.h.{block_id}.mlp.c_fc.weight' in n]

    elif layer_group == 'mlp_proj':
        # MLP 第二层（投影层）
        return [n for n in all_param_names
                if f'transformer.h.{block_id}.mlp.c_proj.weight' in n]

    elif layer_group == 'ln_1':
        # LayerNorm 1 (Attention 前)
        return [n for n in all_param_names
                if f'transformer.h.{block_id}.ln_1.weight' in n]

    elif layer_group == 'ln_2':
        # LayerNorm 2 (MLP 前)
        return [n for n in all_param_names
                if f'transformer.h.{block_id}.ln_2.weight' in n]

    else:
        raise ValueError(f"Unknown layer group: {layer_group}")


def evaluate_loss(model, eval_batches, device):
    """评估模型在多个批次上的平均损失"""
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
    """剪枝指定层次并评估损失"""
    # 获取该层次的重要性得分
    layer_scores = {name: scores[name] for name in layer_params}

    # 展平得分
    flat_scores = torch.cat([s.flatten() for s in layer_scores.values()])

    # 计算阈值
    if flat_scores.numel() > 10_000_000:
        sample_size = 10_000_000
        indices = torch.randperm(flat_scores.numel())[:sample_size]
        sampled_scores = flat_scores[indices]
        threshold = torch.quantile(sampled_scores, sparsity)
        del sampled_scores, indices
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

    # 清理内存
    del layer_scores, flat_scores, masks, original_weights
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'loss': avg_loss,
        'pruned_params': pruned_params,
        'total_params': total_params,
    }


def find_lossless_sparsity(results_df, layer_name, block_id, tolerance=0.001):
    """找到无损剪枝的最大稀疏度"""
    layer_results = results_df[
        (results_df['layer'] == layer_name) &
        (results_df['block_id'] == block_id)
    ].sort_values('sparsity')

    lossless_results = layer_results[layer_results['loss_increase'] <= tolerance]

    if len(lossless_results) == 0:
        return 0.0
    else:
        return lossless_results['sparsity'].max()


def save_results(results, output_dir, baseline_loss):
    """保存结果为 CSV 和图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整 CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / 'fine_grained_pruning_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 完整CSV已保存: {csv_path}")

    # 计算每个层次的无损剪枝比例
    lossless_summary = []

    # Embedding层
    emb_lossless = find_lossless_sparsity(df, 'embedding', -1)
    emb_data = df[(df['layer'] == 'embedding') & (df['block_id'] == -1)]
    emb_total_params = emb_data['total_params'].iloc[0] if len(emb_data) > 0 else 0
    lossless_summary.append({
        'layer': 'Embedding',
        'block_id': -1,
        'max_lossless_sparsity': emb_lossless,
        'max_lossless_percent': emb_lossless * 100,
        'total_params': emb_total_params
    })

    # 所有blocks的细粒度层次
    layer_types = ['attn_qkv', 'attn_proj', 'mlp_fc', 'mlp_proj', 'ln_1', 'ln_2']
    for block_id in range(12):
        for layer_type in layer_types:
            lossless_sparsity = find_lossless_sparsity(df, layer_type, block_id)
            layer_data = df[(df['layer'] == layer_type) & (df['block_id'] == block_id)]
            total_params = layer_data['total_params'].iloc[0] if len(layer_data) > 0 else 0
            lossless_summary.append({
                'layer': layer_type,
                'block_id': block_id,
                'max_lossless_sparsity': lossless_sparsity,
                'max_lossless_percent': lossless_sparsity * 100,
                'total_params': total_params
            })

    # 保存无损剪枝摘要
    lossless_df = pd.DataFrame(lossless_summary)
    lossless_csv_path = output_dir / 'lossless_sparsity_summary.csv'
    lossless_df.to_csv(lossless_csv_path, index=False)
    print(f"✓ 无损剪枝摘要已保存: {lossless_csv_path}")

    # 绘制热力图
    plot_lossless_heatmap(lossless_df, output_dir)

    # 绘制对比图
    plot_block_comparison(df, output_dir)

    # 生成分析报告
    generate_analysis_report(lossless_df, output_dir, baseline_loss)

    # 导出参数数量统计
    export_parameter_counts(lossless_df, output_dir)

    # 绘制参数数量可视化
    plot_parameter_counts(lossless_df, output_dir)


def plot_lossless_heatmap(lossless_df, output_dir):
    """绘制无损剪枝比例热力图"""
    blocks_df = lossless_df[lossless_df['block_id'] >= 0]
    pivot_data = blocks_df.pivot(index='block_id', columns='layer', values='max_lossless_percent')

    # 按照逻辑顺序排列列
    column_order = ['attn_qkv', 'attn_proj', 'mlp_fc', 'mlp_proj', 'ln_1', 'ln_2']
    pivot_data = pivot_data[[col for col in column_order if col in pivot_data.columns]]

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Max Lossless Sparsity (%)'},
        vmin=0,
        vmax=90,
        ax=ax
    )

    ax.set_xlabel('Layer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Block ID', fontsize=12, fontweight='bold')
    ax.set_title('Maximum Lossless Sparsity by Block and Layer Type (Fine-Grained)',
                 fontsize=14, fontweight='bold')

    # 更友好的列标签
    ax.set_xticklabels(['Attn QKV', 'Attn Proj', 'MLP FC', 'MLP Proj', 'LN 1', 'LN 2'])

    plt.tight_layout()
    heatmap_path = output_dir / 'lossless_sparsity_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✓ 热力图已保存: {heatmap_path}")
    plt.close()


def plot_block_comparison(df, output_dir):
    """绘制不同block的剪枝敏感度对比"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    layer_types = ['attn_qkv', 'attn_proj', 'mlp_fc', 'mlp_proj', 'ln_1', 'ln_2']
    layer_names = ['Attention QKV', 'Attention Proj', 'MLP FC', 'MLP Proj', 'LayerNorm 1', 'LayerNorm 2']

    for idx, (layer_type, layer_name) in enumerate(zip(layer_types, layer_names)):
        ax = axes[idx]

        for block_id in range(12):
            block_data = df[(df['layer'] == layer_type) & (df['block_id'] == block_id)]
            block_data = block_data.sort_values('sparsity')

            ax.plot(
                block_data['sparsity'] * 100,
                block_data['loss_increase'],
                marker='o',
                label=f'Block {block_id}',
                alpha=0.7,
                markersize=4
            )

        ax.axhline(0, color='black', linestyle='--', linewidth=2, label='Baseline')
        ax.axhline(0.001, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Lossless Threshold')
        ax.set_xlabel('Sparsity (%)', fontsize=11)
        ax.set_ylabel('Loss Increase', fontsize=11)
        ax.set_title(f'{layer_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    comparison_path = output_dir / 'block_comparison.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存: {comparison_path}")
    plt.close()


def generate_analysis_report(lossless_df, output_dir, baseline_loss):
    """生成分析报告"""
    report_path = output_dir / 'ANALYSIS_REPORT.md'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 细粒度无损剪枝分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**基线损失**: {baseline_loss:.4f}\n\n")
        f.write("**分析粒度**: 只测试 weight 参数（排除 bias）\n\n")
        f.write("---\n\n")

        # Embedding层
        f.write("## Embedding Layer\n\n")
        emb_data = lossless_df[lossless_df['block_id'] == -1]
        if len(emb_data) > 0:
            emb_sparsity = emb_data['max_lossless_percent'].values[0]
            f.write(f"**最大无损剪枝比例**: {emb_sparsity:.1f}%\n\n")

        f.write("---\n\n")

        # 各个block的统计
        f.write("## Transformer Blocks (0-11)\n\n")

        blocks_df = lossless_df[lossless_df['block_id'] >= 0]

        layer_configs = [
            ('attn_qkv', 'Attention QKV (c_attn.weight)'),
            ('attn_proj', 'Attention Projection (c_proj.weight)'),
            ('mlp_fc', 'MLP First Layer (c_fc.weight)'),
            ('mlp_proj', 'MLP Projection (c_proj.weight)'),
            ('ln_1', 'LayerNorm 1 (ln_1.weight)'),
            ('ln_2', 'LayerNorm 2 (ln_2.weight)'),
        ]

        for layer_type, layer_name in layer_configs:
            layer_data = blocks_df[blocks_df['layer'] == layer_type]

            f.write(f"### {layer_name}\n\n")
            f.write("| Block ID | Max Lossless Sparsity (%) |\n")
            f.write("|----------|---------------------------|\n")

            for _, row in layer_data.iterrows():
                f.write(f"| {int(row['block_id'])} | {row['max_lossless_percent']:.1f}% |\n")

            mean_sparsity = layer_data['max_lossless_percent'].mean()
            std_sparsity = layer_data['max_lossless_percent'].std()
            min_sparsity = layer_data['max_lossless_percent'].min()
            max_sparsity = layer_data['max_lossless_percent'].max()

            f.write(f"\n**统计信息**:\n")
            f.write(f"- 平均: {mean_sparsity:.1f}%\n")
            f.write(f"- 标准差: {std_sparsity:.1f}%\n")
            f.write(f"- 最小: {min_sparsity:.1f}%\n")
            f.write(f"- 最大: {max_sparsity:.1f}%\n\n")

        f.write("---\n\n")

        # 关键发现
        f.write("## 关键发现\n\n")

        for layer_type, layer_name in layer_configs:
            layer_data = blocks_df[blocks_df['layer'] == layer_type]
            if len(layer_data) > 0:
                max_row = layer_data.loc[layer_data['max_lossless_percent'].idxmax()]
                min_row = layer_data.loc[layer_data['max_lossless_percent'].idxmin()]

                f.write(f"### {layer_name}\n")
                f.write(f"- **最耐剪枝**: Block {int(max_row['block_id'])} ({max_row['max_lossless_percent']:.1f}%)\n")
                f.write(f"- **最敏感**: Block {int(min_row['block_id'])} ({min_row['max_lossless_percent']:.1f}%)\n\n")

        # 对比分析
        f.write("---\n\n")
        f.write("## 层次对比分析\n\n")

        # Attention: QKV vs Proj
        f.write("### Attention 层对比\n\n")
        attn_qkv_data = blocks_df[blocks_df['layer'] == 'attn_qkv']
        attn_proj_data = blocks_df[blocks_df['layer'] == 'attn_proj']
        f.write(f"- **QKV 平均无损剪枝**: {attn_qkv_data['max_lossless_percent'].mean():.1f}%\n")
        f.write(f"- **Proj 平均无损剪枝**: {attn_proj_data['max_lossless_percent'].mean():.1f}%\n\n")

        # MLP: FC vs Proj
        f.write("### MLP 层对比\n\n")
        mlp_fc_data = blocks_df[blocks_df['layer'] == 'mlp_fc']
        mlp_proj_data = blocks_df[blocks_df['layer'] == 'mlp_proj']
        f.write(f"- **FC 平均无损剪枝**: {mlp_fc_data['max_lossless_percent'].mean():.1f}%\n")
        f.write(f"- **Proj 平均无损剪枝**: {mlp_proj_data['max_lossless_percent'].mean():.1f}%\n\n")

    print(f"✓ 分析报告已保存: {report_path}")


def export_parameter_counts(lossless_df, output_dir):
    """导出参数数量统计到 CSV"""
    output_dir = Path(output_dir)

    # 创建参数数量统计表
    param_counts = []

    # Embedding层
    emb_data = lossless_df[lossless_df['block_id'] == -1]
    if len(emb_data) > 0:
        param_counts.append({
            'layer': 'Embedding',
            'block_id': -1,
            'total_params': int(emb_data['total_params'].iloc[0]),
            'max_lossless_sparsity': emb_data['max_lossless_sparsity'].iloc[0],
            'lossless_prunable_params': int(emb_data['total_params'].iloc[0] * emb_data['max_lossless_sparsity'].iloc[0]),
            'lossless_remaining_params': int(emb_data['total_params'].iloc[0] * (1 - emb_data['max_lossless_sparsity'].iloc[0]))
        })

    # 所有blocks的细粒度层次
    blocks_df = lossless_df[lossless_df['block_id'] >= 0]
    layer_types = ['attn_qkv', 'attn_proj', 'mlp_fc', 'mlp_proj', 'ln_1', 'ln_2']

    for block_id in range(12):
        for layer_type in layer_types:
            layer_data = blocks_df[(blocks_df['layer'] == layer_type) & (blocks_df['block_id'] == block_id)]
            if len(layer_data) > 0:
                total = int(layer_data['total_params'].iloc[0])
                sparsity = layer_data['max_lossless_sparsity'].iloc[0]
                param_counts.append({
                    'layer': layer_type,
                    'block_id': int(block_id),
                    'total_params': total,
                    'max_lossless_sparsity': sparsity,
                    'lossless_prunable_params': int(total * sparsity),
                    'lossless_remaining_params': int(total * (1 - sparsity))
                })

    # 保存为 CSV
    param_df = pd.DataFrame(param_counts)
    param_csv_path = output_dir / 'parameter_counts.csv'
    param_df.to_csv(param_csv_path, index=False)
    print(f"✓ 参数数量统计已保存: {param_csv_path}")

    # 计算并保存汇总统计
    summary_stats = []

    # 按层次类型汇总
    for layer_type in layer_types:
        layer_data = param_df[param_df['layer'] == layer_type]
        if len(layer_data) > 0:
            summary_stats.append({
                'layer_type': layer_type,
                'total_params': int(layer_data['total_params'].sum()),
                'total_lossless_prunable': int(layer_data['lossless_prunable_params'].sum()),
                'total_lossless_remaining': int(layer_data['lossless_remaining_params'].sum()),
                'avg_sparsity': layer_data['max_lossless_sparsity'].mean()
            })

    # 总体统计
    blocks_data = param_df[param_df['block_id'] >= 0]
    summary_stats.append({
        'layer_type': 'ALL_BLOCKS',
        'total_params': int(blocks_data['total_params'].sum()),
        'total_lossless_prunable': int(blocks_data['lossless_prunable_params'].sum()),
        'total_lossless_remaining': int(blocks_data['lossless_remaining_params'].sum()),
        'avg_sparsity': blocks_data['max_lossless_sparsity'].mean()
    })

    summary_df = pd.DataFrame(summary_stats)
    summary_csv_path = output_dir / 'parameter_counts_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✓ 参数数量汇总已保存: {summary_csv_path}")


def plot_parameter_counts(lossless_df, output_dir):
    """绘制参数数量可视化图表"""
    output_dir = Path(output_dir)

    blocks_df = lossless_df[lossless_df['block_id'] >= 0]
    layer_types = ['attn_qkv', 'attn_proj', 'mlp_fc', 'mlp_proj', 'ln_1', 'ln_2']
    layer_names = ['Attn QKV', 'Attn Proj', 'MLP FC', 'MLP Proj', 'LN 1', 'LN 2']

    # 图1: 每个层次类型的总参数数量（堆叠柱状图）
    fig, ax = plt.subplots(figsize=(14, 8))

    layer_totals = []
    for layer_type in layer_types:
        layer_data = blocks_df[blocks_df['layer'] == layer_type]
        total = layer_data['total_params'].sum()
        layer_totals.append(total)

    colors = plt.cm.Set3(range(len(layer_types)))
    bars = ax.bar(layer_names, layer_totals, color=colors, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar, total in zip(bars, layer_totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(total):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Total Parameters', fontsize=13, fontweight='bold')
    ax.set_xlabel('Layer Type', fontsize=13, fontweight='bold')
    ax.set_title('Total Parameter Count by Layer Type (All 12 Blocks)',
                 fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    param_bar_path = output_dir / 'parameter_counts_by_layer.png'
    plt.savefig(param_bar_path, dpi=300, bbox_inches='tight')
    print(f"✓ 参数数量柱状图已保存: {param_bar_path}")
    plt.close()

    # 图2: 可剪枝参数数量热力图（每个block的每个层）
    # 计算可剪枝参数数量 = total_params * max_lossless_sparsity
    blocks_df_copy = blocks_df.copy()
    blocks_df_copy['prunable_params'] = blocks_df_copy['total_params'] * blocks_df_copy['max_lossless_sparsity']

    fig, ax = plt.subplots(figsize=(12, 8))

    pivot_data = blocks_df_copy.pivot(index='block_id', columns='layer', values='prunable_params')
    column_order = ['attn_qkv', 'attn_proj', 'mlp_fc', 'mlp_proj', 'ln_1', 'ln_2']
    pivot_data = pivot_data[[col for col in column_order if col in pivot_data.columns]]

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Lossless Prunable Parameter Count'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_xlabel('Layer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Block ID', fontsize=12, fontweight='bold')
    ax.set_title('Lossless Prunable Parameter Count by Block and Layer Type',
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(['Attn QKV', 'Attn Proj', 'MLP FC', 'MLP Proj', 'LN 1', 'LN 2'])

    plt.tight_layout()
    param_heatmap_path = output_dir / 'lossless_prunable_params_heatmap.png'
    plt.savefig(param_heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✓ 可剪枝参数数量热力图已保存: {param_heatmap_path}")
    plt.close()

    # 图3: 可剪枝参数 vs 保留参数（堆叠柱状图）
    fig, ax = plt.subplots(figsize=(14, 8))

    prunable_counts = []
    remaining_counts = []

    for layer_type in layer_types:
        layer_data = blocks_df[blocks_df['layer'] == layer_type]
        total = layer_data['total_params'].sum()
        avg_sparsity = layer_data['max_lossless_sparsity'].mean()
        prunable = total * avg_sparsity
        remaining = total * (1 - avg_sparsity)
        prunable_counts.append(prunable)
        remaining_counts.append(remaining)

    x = range(len(layer_names))
    width = 0.6

    bars1 = ax.bar(x, remaining_counts, width, label='Remaining (Lossless)',
                   color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, prunable_counts, width, bottom=remaining_counts,
                   label='Prunable (Lossless)', color='coral', edgecolor='black', linewidth=1.5)

    # 添加总数标签
    for i, (prunable, remaining) in enumerate(zip(prunable_counts, remaining_counts)):
        total = prunable + remaining
        ax.text(i, total, f'{int(total):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Parameter Count', fontsize=13, fontweight='bold')
    ax.set_xlabel('Layer Type', fontsize=13, fontweight='bold')
    ax.set_title('Lossless Prunable vs Remaining Parameters by Layer Type',
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    param_stack_path = output_dir / 'parameter_counts_prunable_vs_remaining.png'
    plt.savefig(param_stack_path, dpi=300, bbox_inches='tight')
    print(f"✓ 可剪枝参数对比图已保存: {param_stack_path}")
    plt.close()

    # 图4: 归一化的可剪枝参数比例热力图
    # 计算所有 blocks 的总参数数量
    total_all_blocks_params = blocks_df['total_params'].sum()

    # 计算每个参数组可剪枝参数占总参数的比例
    blocks_df_normalized = blocks_df.copy()
    blocks_df_normalized['prunable_params'] = blocks_df_normalized['total_params'] * blocks_df_normalized['max_lossless_sparsity']
    blocks_df_normalized['normalized_prunable_ratio'] = (blocks_df_normalized['prunable_params'] / total_all_blocks_params) * 100

    # 导出归一化数据到 CSV
    normalized_csv_data = []
    for _, row in blocks_df_normalized.iterrows():
        normalized_csv_data.append({
            'layer': row['layer'],
            'block_id': int(row['block_id']),
            'total_params': int(row['total_params']),
            'max_lossless_sparsity': row['max_lossless_sparsity'],
            'prunable_params': int(row['prunable_params']),
            'normalized_prunable_ratio_percent': row['normalized_prunable_ratio']
        })

    normalized_df = pd.DataFrame(normalized_csv_data)
    normalized_csv_path = output_dir / 'normalized_prunable_ratios.csv'
    normalized_df.to_csv(normalized_csv_path, index=False)
    print(f"✓ 归一化可剪枝比例已保存: {normalized_csv_path}")

    # 绘制归一化比例热力图
    fig, ax = plt.subplots(figsize=(12, 8))

    pivot_data_normalized = blocks_df_normalized.pivot(index='block_id', columns='layer', values='normalized_prunable_ratio')
    column_order = ['attn_qkv', 'attn_proj', 'mlp_fc', 'mlp_proj', 'ln_1', 'ln_2']
    pivot_data_normalized = pivot_data_normalized[[col for col in column_order if col in pivot_data_normalized.columns]]

    sns.heatmap(
        pivot_data_normalized,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Normalized Prunable Ratio (%)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_xlabel('Layer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Block ID', fontsize=12, fontweight='bold')
    ax.set_title(f'Normalized Prunable Parameter Ratio by Block and Layer Type\n(Relative to Total {total_all_blocks_params:,} Parameters)',
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(['Attn QKV', 'Attn Proj', 'MLP FC', 'MLP Proj', 'LN 1', 'LN 2'])

    plt.tight_layout()
    normalized_heatmap_path = output_dir / 'normalized_prunable_ratios_heatmap.png'
    plt.savefig(normalized_heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✓ 归一化可剪枝比例热力图已保存: {normalized_heatmap_path}")
    plt.close()


def run_fine_grained_pruning_experiment(checkpoint_path, output_dir, device='cuda',
                                        num_steps=100, batch_size=4, seq_length=512,
                                        num_eval_batches=10):
    """运行细粒度剪枝实验"""
    print("=" * 80)
    print("细粒度无损剪枝分析（只测试 weight 参数）")
    print("=" * 80)
    print(f"检查点: {checkpoint_path}")
    print(f"梯度累积步数: {num_steps}")
    print(f"评估批次数: {num_eval_batches}")
    print("=" * 80)

    # 加载模型和数据
    print("\n加载模型...")
    model = get_gpt2_small(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✓ 模型已加载")

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

    # 强制垃圾回收
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # 定义实验组
    sparsities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    # 运行实验
    print("\n" + "=" * 80)
    print("开始剪枝实验")
    print("=" * 80)

    results = []
    all_param_names = list(scores.keys())

    # 定期保存中间结果
    def save_intermediate(results_list, output_path):
        if len(results_list) > 0:
            df = pd.DataFrame(results_list)
            df.to_csv(output_path, index=False)

    intermediate_csv = Path(output_dir) / 'intermediate_results.csv'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 测试Embedding层
    print(f"\n{'='*80}")
    print("测试 Embedding Layer")
    print(f"{'='*80}")

    layer_params = select_layer_params('embedding', -1, all_param_names)
    print(f"  参数: {layer_params}")

    for sparsity in sparsities:
        result = prune_and_evaluate(
            model, layer_params, scores, sparsity, eval_batches, device
        )

        result['layer'] = 'embedding'
        result['layer_name'] = 'Embedding Layer'
        result['block_id'] = -1
        result['sparsity'] = sparsity
        result['baseline_loss'] = baseline_loss
        result['loss_increase'] = result['loss'] - baseline_loss

        results.append(result)

        print(f"  稀疏度 {sparsity*100:5.1f}%: "
              f"loss={result['loss']:.4f}, "
              f"Δloss={result['loss_increase']:+.4f}")

    save_intermediate(results, intermediate_csv)

    # 测试所有12个blocks的细粒度层次
    layer_configs = [
        ('attn_qkv', 'Attention QKV'),
        ('attn_proj', 'Attention Proj'),
        ('mlp_fc', 'MLP FC'),
        ('mlp_proj', 'MLP Proj'),
        ('ln_1', 'LayerNorm 1'),
        ('ln_2', 'LayerNorm 2'),
    ]

    for block_id in range(12):
        print(f"\n{'='*80}")
        print(f"测试 Block {block_id}")
        print(f"{'='*80}")

        for layer_type, layer_display_name in layer_configs:
            print(f"\n  [Block {block_id}] {layer_display_name}")
            layer_params = select_layer_params(layer_type, block_id, all_param_names)

            if len(layer_params) == 0:
                print(f"    警告: 未找到参数")
                continue

            print(f"    参数: {layer_params}")

            for sparsity in sparsities:
                result = prune_and_evaluate(
                    model, layer_params, scores, sparsity, eval_batches, device
                )

                result['layer'] = layer_type
                result['layer_name'] = f'Block {block_id} - {layer_display_name}'
                result['block_id'] = block_id
                result['sparsity'] = sparsity
                result['baseline_loss'] = baseline_loss
                result['loss_increase'] = result['loss'] - baseline_loss

                results.append(result)

                print(f"    稀疏度 {sparsity*100:5.1f}%: "
                      f"loss={result['loss']:.4f}, "
                      f"Δloss={result['loss_increase']:+.4f}")

        # 每个block完成后保存并清理内存
        save_intermediate(results, intermediate_csv)
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    # 保存最终结果
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)
    save_results(results, output_dir, baseline_loss)

    print("\n" + "=" * 80)
    print("实验完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description='细粒度无损剪枝分析（只测试 weight 参数）')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='检查点路径')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='梯度累积步数')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='批次大小')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='序列长度')
    parser.add_argument('--num_eval_batches', type=int, default=10,
                        help='评估批次数')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备')
    parser.add_argument('--output_dir', type=str,
                        default='results/fine_grained_pruning_analysis',
                        help='输出目录')

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = 'cpu'

    run_fine_grained_pruning_experiment(
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

