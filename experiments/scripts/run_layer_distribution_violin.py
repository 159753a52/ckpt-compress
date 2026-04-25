"""
§9.4 NLP 模型层间特性分析 — Score 分布异质性 Violin Plot

展示不同层类型的 damage score 分布差异，证明 Uniform 分配不合理。

运行示例:
    python experiments/scripts/run_layer_distribution_violin.py \
        --model gpt2-medium --dataset wikitext103 \
        --importance second-order-hvp --device cuda
"""

import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

import sys
from pathlib import Path
import argparse
import copy
import json
import re
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lib.models import load_model, get_model_type
from experiments.lib.data import get_data_loaders, cache_batches
from experiments.lib.importance_compare.scoring import compute_scores_by_method


# ============================================================
#  Layer type classification
# ============================================================

def classify_layer(name, model_type):
    """将参数名映射到层类型 + 块编号。"""
    name_lower = name.lower()

    # GPT-2 / Pythia
    if model_type in ('gpt2', 'pythia'):
        block_match = re.search(r'\.h\.(\d+)\.|\.layers\.(\d+)\.', name)
        block_id = int(block_match.group(1) or block_match.group(2)) if block_match else -1

        if 'wte' in name_lower or 'wpe' in name_lower or 'embed' in name_lower:
            return 'Embedding', block_id
        elif 'ln' in name_lower or 'layernorm' in name_lower or 'layer_norm' in name_lower:
            return 'LayerNorm', block_id
        elif 'c_attn' in name_lower or 'q_proj' in name_lower or 'k_proj' in name_lower or 'v_proj' in name_lower:
            return 'Attention-QKV', block_id
        elif 'c_proj' in name_lower and 'attn' in name_lower:
            return 'Attention-Out', block_id
        elif 'c_fc' in name_lower or 'fc1' in name_lower or 'mlp.c_fc' in name_lower or 'up_proj' in name_lower:
            return 'MLP-Up', block_id
        elif 'c_proj' in name_lower or 'fc2' in name_lower or 'down_proj' in name_lower:
            return 'MLP-Down', block_id

    # ViT
    elif model_type == 'vit':
        block_match = re.search(r'\.(\d+)\.', name)
        block_id = int(block_match.group(1)) if block_match else -1

        if 'patch_embed' in name_lower or 'cls_token' in name_lower or 'pos_embed' in name_lower:
            return 'Embedding', block_id
        elif 'norm' in name_lower or 'ln' in name_lower:
            return 'LayerNorm', block_id
        elif 'qkv' in name_lower or 'q_proj' in name_lower or 'k_proj' in name_lower or 'v_proj' in name_lower:
            return 'Attention-QKV', block_id
        elif 'proj' in name_lower and 'attn' in name_lower:
            return 'Attention-Out', block_id
        elif 'fc1' in name_lower or 'mlp.0' in name_lower:
            return 'MLP-Up', block_id
        elif 'fc2' in name_lower or 'mlp.3' in name_lower:
            return 'MLP-Down', block_id

    return 'Other', -1


def build_score_dataframe(scores, model_type, max_samples_per_layer=5000):
    """构建用于绘图的 DataFrame。"""
    rows = []
    for name, score in scores.items():
        layer_type, block_id = classify_layer(name, model_type)
        data = score.flatten().float().cpu().numpy()
        data_pos = data[data > 1e-12]
        if len(data_pos) == 0:
            continue

        # 采样防止数据过大
        if len(data_pos) > max_samples_per_layer:
            idx = np.random.choice(len(data_pos), max_samples_per_layer, replace=False)
            data_pos = data_pos[idx]

        for val in data_pos:
            rows.append({
                'layer_name': name,
                'layer_type': layer_type,
                'block_id': block_id,
                'log_score': np.log10(val + 1e-20),
                'score': val,
            })
    return pd.DataFrame(rows)


# ============================================================
#  Plotting
# ============================================================

TYPE_COLORS = {
    'Embedding': '#e74c3c',      # 红
    'Attention-QKV': '#3498db',   # 蓝
    'Attention-Out': '#2980b9',   # 深蓝
    'MLP-Up': '#e67e22',          # 橙
    'MLP-Down': '#d35400',        # 深橙
    'LayerNorm': '#2ecc71',       # 绿
    'Other': '#95a5a6',           # 灰
}


def plot_violin_by_type(df, output_path, model_name):
    """按层类型绘制 violin plot。"""
    type_order = ['Embedding', 'Attention-QKV', 'Attention-Out',
                  'MLP-Up', 'MLP-Down', 'LayerNorm']
    present_types = [t for t in type_order if t in df['layer_type'].unique()]

    fig, ax = plt.subplots(figsize=(12, 6))

    try:
        import seaborn as sns
        palette = [TYPE_COLORS.get(t, '#95a5a6') for t in present_types]
        sns.violinplot(data=df[df['layer_type'].isin(present_types)],
                       x='layer_type', y='log_score',
                       order=present_types, palette=palette,
                       cut=0, inner='quartile', ax=ax)
    except ImportError:
        # Fallback: box plot
        box_data = [df[df['layer_type'] == t]['log_score'].values for t in present_types]
        ax.boxplot(box_data, labels=present_types)

    ax.set_xlabel('Layer Type', fontsize=13)
    ax.set_ylabel('log₁₀(Damage Score)', fontsize=13)
    ax.set_title(f'Score Distribution Heterogeneity — {model_name}', fontsize=14)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_violin_by_block(df, output_path, model_name, n_blocks_to_show=6):
    """选取代表性块，按层类型展开。"""
    blocks = sorted(df[df['block_id'] >= 0]['block_id'].unique())
    if len(blocks) == 0:
        return

    # 选取首/中/尾块
    indices = np.linspace(0, len(blocks) - 1, min(n_blocks_to_show, len(blocks)), dtype=int)
    selected_blocks = [blocks[i] for i in indices]

    subset = df[df['block_id'].isin(selected_blocks)]
    subset = subset.copy()
    subset['block_label'] = subset['block_id'].apply(lambda x: f'Block {x}')

    fig, axes = plt.subplots(1, len(selected_blocks), figsize=(4 * len(selected_blocks), 5),
                              sharey=True)
    if len(selected_blocks) == 1:
        axes = [axes]

    type_order = ['Attention-QKV', 'Attention-Out', 'MLP-Up', 'MLP-Down', 'LayerNorm']

    for ax, block_id in zip(axes, selected_blocks):
        block_data = subset[subset['block_id'] == block_id]
        present = [t for t in type_order if t in block_data['layer_type'].unique()]
        if not present:
            continue

        try:
            import seaborn as sns
            palette = [TYPE_COLORS.get(t, '#95a5a6') for t in present]
            sns.violinplot(data=block_data[block_data['layer_type'].isin(present)],
                           x='layer_type', y='log_score',
                           order=present, palette=palette,
                           cut=0, inner='quartile', ax=ax)
        except ImportError:
            box_data = [block_data[block_data['layer_type'] == t]['log_score'].values
                        for t in present]
            ax.boxplot(box_data, labels=present)

        ax.set_title(f'Block {block_id}', fontsize=12)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('log₁₀(Damage Score)', fontsize=12)
    fig.suptitle(f'Per-Block Score Distribution — {model_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def compute_statistics(df):
    """计算每层类型的统计指标。"""
    stats = df.groupby('layer_type').agg(
        count=('score', 'size'),
        mean_log=('log_score', 'mean'),
        std_log=('log_score', 'std'),
        median_log=('log_score', 'median'),
        iqr_log=('log_score', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    ).round(4)
    return stats


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='§9.4 层间 Score 分布异质性分析')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--importance', type=str, default='second-order-hvp')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--hvp_batches', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str,
                        default='results/paper_results/layer_violin')
    args = parser.parse_args()

    print("=" * 70)
    print("§9.4 层间 Score 分布异质性分析")
    print(f"模型: {args.model} | 数据集: {args.dataset}")
    print("=" * 70)

    model, model_family = load_model(args.model, pretrained=True,
                          checkpoint_path=args.checkpoint, device='cpu',
                          dataset_name=args.dataset)
    model_type = get_model_type(args.model)
    train_loader, _, task_type = get_data_loaders(
        args.model, args.dataset, args.batch_size, args.seq_length)
    cached_train = cache_batches(train_loader, args.num_steps, task_type)

    # 计算 scores
    print("\n[1/3] 计算 importance scores...")
    model_s = copy.deepcopy(model).to(args.device)
    score_cache = compute_scores_by_method(
        model_s, cached_train, task_type,
        methods=[args.importance], alpha=args.alpha,
        hvp_batches=args.hvp_batches,
        model_family=model_family,
    )
    scores = score_cache[args.importance]
    del model_s
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 构建 DataFrame
    print("\n[2/3] 构建分布数据...")
    df = build_score_dataframe(scores, model_type)
    print(f"  总样本数: {len(df)}, 层类型: {df['layer_type'].nunique()}")

    # 统计
    stats = compute_statistics(df)
    print("\n层类型统计:")
    print(stats.to_string())

    # 绘图
    print("\n[3/3] 生成可视化...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_violin_by_type(
        df, output_dir / f'{args.model}_violin_by_type.png', args.model)
    plot_violin_by_block(
        df, output_dir / f'{args.model}_violin_by_block.png', args.model)

    # 保存数据
    stats.to_csv(output_dir / f'{args.model}_layer_stats.csv')
    with open(output_dir / f'{args.model}_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\n完成！")


if __name__ == '__main__':
    main()
