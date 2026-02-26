"""
可视化 GPT-2 Small 和 BERT-Large 模型的参数量分布。

运行命令:
    # GPT-2 Small
    python experiments/scripts/visualize_model_structure.py --model gpt2_small

    # BERT-Large
    python experiments/scripts/visualize_model_structure.py --model bert_large
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.models.bert import get_bert_large


def get_model(model_name: str):
    """获取模型"""
    if model_name == 'gpt2_small':
        return get_gpt2_small(pretrained=False)
    elif model_name == 'bert_large':
        return get_bert_large(pretrained=False, local_files_only=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def analyze_model_structure(model_name: str = 'gpt2_small'):
    """分析模型结构并返回统计数据。

    Args:
        model_name: 模型名称 ('gpt2_small' 或 'bert_large')
    """
    model = get_model(model_name)

    param_stats = {}
    total_params = 0

    for name, param in model.named_parameters():
        n_params = param.numel()
        total_params += n_params
        param_stats[name] = n_params

    # 确定模型类型和参数
    if model_name == 'gpt2_small':
        num_blocks = 12
        block_prefix = 'transformer.h.'
        embedding_keywords = ['wte.weight', 'wpe.weight']
        ln_keywords = ['ln_', 'ln_f']
    elif model_name == 'bert_large':
        num_blocks = 24
        block_prefix = 'bert.encoder.layer.'
        embedding_keywords = ['bert.embeddings']
        ln_keywords = ['LayerNorm']
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 按类型分组
    embedding_params = 0
    layernorm_params = 0
    attention_params = 0
    mlp_params = 0

    for name, n_params in param_stats.items():
        # 检查是否是 embedding
        is_embedding = any(kw in name for kw in embedding_keywords)
        # 检查是否是 layernorm
        is_layernorm = any(kw in name for kw in ln_keywords)

        if is_embedding:
            embedding_params += n_params
        elif is_layernorm:
            layernorm_params += n_params
        elif 'attention' in name or 'attn' in name:
            attention_params += n_params
        elif 'mlp' in name or 'intermediate' in name or 'output.dense' in name:
            mlp_params += n_params

    # 按 Block 分组
    block_params = []
    for block_idx in range(num_blocks):
        block_total = 0
        for name, n_params in param_stats.items():
            if f'{block_prefix}{block_idx}.' in name:
                block_total += n_params
        block_params.append(block_total)

    # 单个 Block 内部结构
    if model_name == 'gpt2_small':
        block0_structure = {
            'LayerNorm': 0,
            'Attention QKV': 0,
            'Attention Out': 0,
            'MLP FC1': 0,
            'MLP FC2': 0,
        }

        for name, n_params in param_stats.items():
            if 'transformer.h.0.' in name:
                if 'ln_' in name:
                    block0_structure['LayerNorm'] += n_params
                elif 'attn.c_attn' in name:
                    block0_structure['Attention QKV'] += n_params
                elif 'attn.c_proj' in name:
                    block0_structure['Attention Out'] += n_params
                elif 'mlp.c_fc' in name:
                    block0_structure['MLP FC1'] += n_params
                elif 'mlp.c_proj' in name:
                    block0_structure['MLP FC2'] += n_params

    else:  # bert_large
        block0_structure = {
            'LayerNorm (Attention)': 0,
            'Attention Q': 0,
            'Attention K': 0,
            'Attention V': 0,
            'Attention Output': 0,
            'LayerNorm (FFN)': 0,
            'FFN Intermediate': 0,
            'FFN Output': 0,
        }

        for name, n_params in param_stats.items():
            if 'bert.encoder.layer.0.' in name:
                if 'attention.output.LayerNorm' in name:
                    block0_structure['LayerNorm (Attention)'] += n_params
                elif 'attention.self.query' in name:
                    block0_structure['Attention Q'] += n_params
                elif 'attention.self.key' in name:
                    block0_structure['Attention K'] += n_params
                elif 'attention.self.value' in name:
                    block0_structure['Attention V'] += n_params
                elif 'attention.output.dense' in name:
                    block0_structure['Attention Output'] += n_params
                elif 'output.LayerNorm' in name:
                    block0_structure['LayerNorm (FFN)'] += n_params
                elif 'intermediate.dense' in name:
                    block0_structure['FFN Intermediate'] += n_params
                elif 'output.dense' in name:
                    block0_structure['FFN Output'] += n_params

    return {
        'total_params': total_params,
        'embedding_params': embedding_params,
        'layernorm_params': layernorm_params,
        'attention_params': attention_params,
        'mlp_params': mlp_params,
        'block_params': block_params,
        'block0_structure': block0_structure,
        'num_blocks': num_blocks,
        'model_name': model_name,
    }


def plot_model_structure(stats, output_dir):
    """Plot model structure visualization charts.

    Args:
        stats: 模型统计数据
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = stats['model_name']
    num_blocks = stats['num_blocks']
    display_name = 'GPT-2 Small' if model_name == 'gpt2_small' else 'BERT-Large'

    # Figure 1: Overall module distribution (pie chart)
    fig, ax = plt.subplots(figsize=(10, 8))

    # 计算百分比
    embedding_pct = stats['embedding_params'] / stats['total_params'] * 100
    mlp_pct = stats['mlp_params'] / stats['total_params'] * 100
    attention_pct = stats['attention_params'] / stats['total_params'] * 100
    ln_pct = stats['layernorm_params'] / stats['total_params'] * 100

    labels = [
        f'Embedding\n({embedding_pct:.2f}%)',
        f'MLP\n({mlp_pct:.2f}%)',
        f'Attention\n({attention_pct:.2f}%)',
        f'LayerNorm\n({ln_pct:.2f}%)'
    ]
    sizes = [
        stats['embedding_params'],
        stats['mlp_params'],
        stats['attention_params'],
        stats['layernorm_params'],
    ]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    explode = (0.05, 0.05, 0.05, 0.1)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=explode,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    # Add parameter count annotations
    for i, (wedge, size) in enumerate(zip(wedges, sizes)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 1.3 * np.cos(np.radians(angle))
        y = 1.3 * np.sin(np.radians(angle))
        ax.text(x, y, f'{size:,}', ha='center', va='center', fontsize=10, style='italic')

    ax.set_title(f'{display_name} Parameter Distribution\nTotal Parameters: {stats["total_params"]:,}',
                 fontsize=16, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_structure_pie.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'model_structure_pie.png'}")
    plt.close()

    # Figure 2: Block distribution (bar chart)
    fig, ax = plt.subplots(figsize=(14, 6))

    x_labels = ['Embedding'] + [f'Block {i}' for i in range(num_blocks)] + ['Final LN']
    x_pos = np.arange(len(x_labels))

    heights = [stats['embedding_params']] + stats['block_params'] + [stats['layernorm_params']]
    colors_bar = ['#ff9999'] + ['#66b3ff'] * num_blocks + ['#ffcc99']

    bars = ax.bar(x_pos, heights, color=colors_bar, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, height in zip(bars, heights):
        if height > 1000000:
            label = f'{height/1e6:.1f}M'
        elif height > 1000:
            label = f'{height/1e3:.1f}K'
        else:
            label = f'{height}'
        ax.text(bar.get_x() + bar.get_width()/2, height + 1e6, label,
                ha='center', va='bottom', fontsize=9, weight='bold')

    ax.set_xlabel('Module', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Parameters', fontsize=12, weight='bold')
    ax.set_title(f'{display_name} Parameter Distribution by Module', fontsize=14, weight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(heights) * 1.15)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff9999', edgecolor='black', label='Embedding'),
        Patch(facecolor='#66b3ff', edgecolor='black', label='Transformer Block'),
        Patch(facecolor='#ffcc99', edgecolor='black', label='Final LayerNorm'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_structure_bar.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'model_structure_bar.png'}")
    plt.close()

    # Figure 3: Single block internal structure (horizontal bar chart)
    fig, ax = plt.subplots(figsize=(12, 6) if model_name == 'bert_large' else (10, 6))

    components = list(stats['block0_structure'].keys())
    values = list(stats['block0_structure'].values())

    # Sort by size
    sorted_indices = np.argsort(values)[::-1]
    components = [components[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]

    colors_comp = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

    y_pos = np.arange(len(components))
    bars = ax.barh(y_pos, values, color=colors_comp, edgecolor='black', linewidth=1.5)

    # Add value and percentage labels
    block_total = sum(values)
    for i, (bar, value) in enumerate(zip(bars, values)):
        percentage = value / block_total * 100
        label = f'{value:,} ({percentage:.1f}%)'
        ax.text(value + 50000, bar.get_y() + bar.get_height()/2, label,
                ha='left', va='center', fontsize=10, weight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(components, fontsize=11)
    ax.set_xlabel('Number of Parameters', fontsize=12, weight='bold')
    ax.set_title(f'Single Transformer Block Internal Structure (Block 0)\nTotal Parameters: {block_total:,}',
                 fontsize=14, weight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(values) * 1.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'block_structure.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'block_structure.png'}")
    plt.close()

    # Figure 4: Comprehensive comparison (2x2 subplots)
    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: Overall pie chart
    ax1 = plt.subplot(2, 2, 1)
    labels_simple = ['Embedding', 'MLP', 'Attention', 'LayerNorm']
    sizes_simple = [
        stats['embedding_params'],
        stats['mlp_params'],
        stats['attention_params'],
        stats['layernorm_params'],
    ]
    colors_simple = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    wedges, texts, autotexts = ax1.pie(
        sizes_simple,
        labels=labels_simple,
        colors=colors_simple,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10, 'weight': 'bold'}
    )
    ax1.set_title('Overall Module Distribution', fontsize=12, weight='bold')

    # Subplot 2: Block distribution
    ax2 = plt.subplot(2, 2, 2)
    block_indices = np.arange(num_blocks)
    ax2.bar(block_indices, stats['block_params'], color='#66b3ff', edgecolor='black')
    ax2.set_xlabel('Block Index', fontsize=10, weight='bold')
    ax2.set_ylabel('Number of Parameters', fontsize=10, weight='bold')
    ax2.set_title(f'{num_blocks} Transformer Blocks Parameter Distribution', fontsize=12, weight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticks(block_indices)

    # Subplot 3: Block internal structure
    ax3 = plt.subplot(2, 2, 3)
    components_sorted = list(stats['block0_structure'].keys())
    values_sorted = list(stats['block0_structure'].values())
    sorted_idx = np.argsort(values_sorted)[::-1]
    components_sorted = [components_sorted[i] for i in sorted_idx]
    values_sorted = [values_sorted[i] for i in sorted_idx]

    y_pos_sorted = np.arange(len(components_sorted))
    ax3.barh(y_pos_sorted, values_sorted, color=colors_comp, edgecolor='black')
    ax3.set_yticks(y_pos_sorted)
    ax3.set_yticklabels(components_sorted, fontsize=9)
    ax3.set_xlabel('Number of Parameters', fontsize=10, weight='bold')
    ax3.set_title('Single Block Internal Structure', fontsize=12, weight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # Subplot 4: Parameter statistics table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    # 计算百分比
    embed_pct = stats['embedding_params'] / stats['total_params'] * 100
    mlp_pct = stats['mlp_params'] / stats['total_params'] * 100
    attn_pct = stats['attention_params'] / stats['total_params'] * 100
    ln_pct = stats['layernorm_params'] / stats['total_params'] * 100

    table_data = [
        ['Module', 'Parameters', 'Percentage'],
        ['Embedding', f"{stats['embedding_params']:,}", f'{embed_pct:.2f}%'],
        [f'MLP ({num_blocks} blocks)', f"{stats['mlp_params']:,}", f'{mlp_pct:.2f}%'],
        [f'Attention ({num_blocks} blocks)', f"{stats['attention_params']:,}", f'{attn_pct:.2f}%'],
        ['LayerNorm', f"{stats['layernorm_params']:,}", f'{ln_pct:.2f}%'],
        ['', '', ''],
        ['Total', f"{stats['total_params']:,}", '100.00%'],
    ]

    table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                      colWidths=[0.4, 0.35, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Set header style
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Set total row style
    for i in range(3):
        table[(6, i)].set_facecolor('#FFC107')
        table[(6, i)].set_text_props(weight='bold')

    ax4.set_title('Parameter Statistics Table', fontsize=12, weight='bold', pad=20)

    plt.suptitle(f'{display_name} Model Structure Comprehensive Analysis', fontsize=16, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'model_structure_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'model_structure_comprehensive.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='可视化模型结构')
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2_small',
        choices=['gpt2_small', 'bert_large'],
        help='模型选择'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/model_structure_visualization',
        help='输出目录'
    )

    args = parser.parse_args()

    display_name = 'GPT-2 Small' if args.model == 'gpt2_small' else 'BERT-Large'

    print("=" * 80)
    print(f"{display_name} Model Structure Visualization")
    print("=" * 80)

    print("\nAnalyzing model structure...")
    stats = analyze_model_structure(args.model)

    print(f"\nTotal parameters: {stats['total_params']:,}")
    print(f"  - Embedding:  {stats['embedding_params']:,} ({stats['embedding_params']/stats['total_params']*100:.2f}%)")
    print(f"  - MLP:        {stats['mlp_params']:,} ({stats['mlp_params']/stats['total_params']*100:.2f}%)")
    print(f"  - Attention:  {stats['attention_params']:,} ({stats['attention_params']/stats['total_params']*100:.2f}%)")
    print(f"  - LayerNorm:  {stats['layernorm_params']:,} ({stats['layernorm_params']/stats['total_params']*100:.2f}%)")

    print("\nGenerating visualization charts...")
    output_dir = Path(args.output_dir) / args.model
    plot_model_structure(stats, output_dir)

    print("\n" + "=" * 80)
    print("Visualization completed!")
    print(f"Results saved in: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
