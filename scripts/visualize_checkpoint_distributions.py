#!/usr/bin/env python3
"""
可视化检查点参数分布

加载 GPT-2 检查点并绘制：
1. Embedding 层参数分布
2. 第一个 Transformer Block 内部各参数组的分布
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_checkpoint(checkpoint_path: str):
    """加载检查点"""
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 检查点可能包含 'model_state_dict' 或直接是 state_dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"检查点信息:")
            if 'step' in checkpoint:
                print(f"  - Step: {checkpoint['step']}")
            if 'loss' in checkpoint:
                print(f"  - Loss: {checkpoint['loss']:.4f}")
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    print(f"\n状态字典包含 {len(state_dict)} 个参数")
    return state_dict


def get_embedding_params(state_dict):
    """提取 Embedding 层参数"""
    embedding_params = {}

    for key, value in state_dict.items():
        # GPT-2 的 embedding 层
        if 'wte' in key or 'wpe' in key:  # word token embedding, word position embedding
            embedding_params[key] = value

    return embedding_params


def get_first_block_params(state_dict):
    """提取第一个 Transformer Block 的参数"""
    first_block_params = {}

    for key, value in state_dict.items():
        # 第一个 block: transformer.h.0.xxx
        if '.h.0.' in key or 'transformer.h.0.' in key:
            first_block_params[key] = value

    return first_block_params


def categorize_block_params(block_params):
    """将 block 参数按组分类"""
    categories = {
        'LayerNorm 1': {},
        'Attention QKV': {},
        'Attention Proj': {},
        'LayerNorm 2': {},
        'MLP FC1': {},
        'MLP FC2': {},
    }

    for key, value in block_params.items():
        if 'ln_1' in key:
            categories['LayerNorm 1'][key] = value
        elif 'attn.c_attn' in key:  # Q, K, V 投影
            categories['Attention QKV'][key] = value
        elif 'attn.c_proj' in key:  # Attention 输出投影
            categories['Attention Proj'][key] = value
        elif 'ln_2' in key:
            categories['LayerNorm 2'][key] = value
        elif 'mlp.c_fc' in key:  # MLP 第一层
            categories['MLP FC1'][key] = value
        elif 'mlp.c_proj' in key:  # MLP 第二层
            categories['MLP FC2'][key] = value

    return categories


def plot_distribution(ax, tensor, title, color='blue'):
    """绘制单个参数的分布"""
    values = tensor.flatten().numpy()

    # 绘制直方图
    ax.hist(values, bins=100, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)

    # 添加统计信息
    mean = np.mean(values)
    std = np.std(values)
    median = np.median(values)

    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')

    ax.set_title(f'{title}\n(std={std:.4f}, size={values.shape[0]:,})', fontsize=10)
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def visualize_checkpoint(checkpoint_path: str, output_dir: str = './results/checkpoint_visualization'):
    """可视化检查点参数分布"""

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载检查点
    state_dict = load_checkpoint(checkpoint_path)

    # 提取参数
    embedding_params = get_embedding_params(state_dict)
    first_block_params = get_first_block_params(state_dict)

    print(f"\n找到 {len(embedding_params)} 个 Embedding 参数")
    print(f"找到 {len(first_block_params)} 个第一个 Block 的参数")

    # ========== 1. 绘制 Embedding 层分布 ==========
    print("\n绘制 Embedding 层分布...")

    fig, axes = plt.subplots(1, len(embedding_params), figsize=(6*len(embedding_params), 5))
    if len(embedding_params) == 1:
        axes = [axes]

    for idx, (key, tensor) in enumerate(embedding_params.items()):
        short_name = key.split('.')[-2] + '.' + key.split('.')[-1]  # 简化名称
        plot_distribution(axes[idx], tensor, f'Embedding: {short_name}', color='steelblue')
        print(f"  - {key}: {tensor.shape}")

    plt.tight_layout()
    embedding_output = output_path / 'embedding_distribution.png'
    plt.savefig(embedding_output, dpi=300, bbox_inches='tight')
    print(f"保存到: {embedding_output}")
    plt.close()

    # ========== 2. 绘制第一个 Block 的分布 ==========
    print("\n绘制第一个 Block 的参数分布...")

    # 按类别分组
    categorized = categorize_block_params(first_block_params)

    # 统计每个类别的参数数量
    total_params = 0
    for category, params in categorized.items():
        if params:
            print(f"\n{category}:")
            for key, tensor in params.items():
                print(f"  - {key.split('transformer.h.0.')[-1]}: {tensor.shape}")
                total_params += 1

    # 创建子图 (2行3列)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = ['steelblue', 'coral', 'lightgreen', 'gold', 'plum', 'lightcoral']

    for idx, (category, params) in enumerate(categorized.items()):
        if not params:
            axes[idx].text(0.5, 0.5, f'{category}\n(No parameters)',
                          ha='center', va='center', fontsize=12)
            axes[idx].axis('off')
            continue

        # 合并该类别的所有参数
        all_values = []
        param_info = []
        for key, tensor in params.items():
            all_values.append(tensor.flatten().numpy())
            short_name = key.split('transformer.h.0.')[-1]
            param_info.append(f"{short_name}: {tensor.shape}")

        all_values = np.concatenate(all_values)

        # 绘制分布
        axes[idx].hist(all_values, bins=100, alpha=0.7, color=colors[idx],
                      edgecolor='black', linewidth=0.5)

        mean = np.mean(all_values)
        std = np.std(all_values)
        median = np.median(all_values)

        axes[idx].axvline(mean, color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {mean:.4f}')
        axes[idx].axvline(median, color='green', linestyle='--', linewidth=2,
                         label=f'Median: {median:.4f}')

        # 标题包含统计信息
        title = f'{category}\n'
        title += f'std={std:.4f}, size={len(all_values):,}\n'
        title += '\n'.join(param_info)

        axes[idx].set_title(title, fontsize=9)
        axes[idx].set_xlabel('Parameter Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('First Transformer Block - Parameter Distributions', fontsize=16, y=0.995)
    plt.tight_layout()

    block_output = output_path / 'first_block_distribution.png'
    plt.savefig(block_output, dpi=300, bbox_inches='tight')
    print(f"\n保存到: {block_output}")
    plt.close()

    # ========== 3. 绘制详细的参数分布（每个参数单独一个子图）==========
    print("\n绘制详细的参数分布...")

    n_params = len(first_block_params)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for idx, (key, tensor) in enumerate(first_block_params.items()):
        short_name = key.split('transformer.h.0.')[-1]

        # 确定颜色
        if 'ln_1' in key:
            color = 'steelblue'
        elif 'attn.c_attn' in key:
            color = 'coral'
        elif 'attn.c_proj' in key:
            color = 'lightgreen'
        elif 'ln_2' in key:
            color = 'gold'
        elif 'mlp.c_fc' in key:
            color = 'plum'
        elif 'mlp.c_proj' in key:
            color = 'lightcoral'
        else:
            color = 'gray'

        plot_distribution(axes[idx], tensor, short_name, color=color)

    # 隐藏多余的子图
    for idx in range(len(first_block_params), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('First Transformer Block - Detailed Parameter Distributions',
                 fontsize=16, y=0.995)
    plt.tight_layout()

    detailed_output = output_path / 'first_block_detailed_distribution.png'
    plt.savefig(detailed_output, dpi=300, bbox_inches='tight')
    print(f"保存到: {detailed_output}")
    plt.close()

    print(f"\n✅ 所有可视化完成！结果保存在: {output_path}")

    # 打印统计摘要
    print("\n" + "="*60)
    print("参数统计摘要")
    print("="*60)

    print("\nEmbedding 层:")
    for key, tensor in embedding_params.items():
        values = tensor.flatten().numpy()
        print(f"  {key}:")
        print(f"    Shape: {tensor.shape}")
        print(f"    Mean: {np.mean(values):.6f}, Std: {np.std(values):.6f}")
        print(f"    Min: {np.min(values):.6f}, Max: {np.max(values):.6f}")

    print("\n第一个 Block:")
    for category, params in categorized.items():
        if params:
            all_values = np.concatenate([p.flatten().numpy() for p in params.values()])
            print(f"  {category}:")
            print(f"    Total params: {len(all_values):,}")
            print(f"    Mean: {np.mean(all_values):.6f}, Std: {np.std(all_values):.6f}")
            print(f"    Min: {np.min(all_values):.6f}, Max: {np.max(all_values):.6f}")


def main():
    parser = argparse.ArgumentParser(description='可视化检查点参数分布')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/lihongliang/fangzl/ckpt-compress/checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt',
        help='检查点路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/checkpoint_visualization',
        help='输出目录'
    )

    args = parser.parse_args()

    visualize_checkpoint(args.checkpoint, args.output_dir)


if __name__ == '__main__':
    main()
