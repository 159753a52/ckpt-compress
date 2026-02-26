#!/usr/bin/env python3
"""
分析检查点参数的离群值分布

目标：
1. 检测每个权重矩阵中的离群值
2. 统计离群值的比例和分布
3. 导出详细的统计数据到CSV
4. 可视化离群值的影响

离群值检测方法：
- IQR方法：Q1 - 1.5*IQR, Q3 + 1.5*IQR
- Z-score方法：|z| > 3
- 百分位方法：< 0.1% 或 > 99.9%
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, Tuple, List
import seaborn as sns


def load_checkpoint(checkpoint_path: str):
    """加载检查点"""
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    print(f"状态字典包含 {len(state_dict)} 个参数")
    return state_dict


def detect_outliers_iqr(values: np.ndarray, factor: float = 1.5) -> Tuple[np.ndarray, float, float]:
    """使用IQR方法检测离群值"""
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    outliers = (values < lower_bound) | (values > upper_bound)
    return outliers, lower_bound, upper_bound


def detect_outliers_zscore(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """使用Z-score方法检测离群值"""
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return np.zeros(len(values), dtype=bool)

    z_scores = np.abs((values - mean) / std)
    outliers = z_scores > threshold
    return outliers


def detect_outliers_percentile(values: np.ndarray, lower_pct: float = 0.1, upper_pct: float = 99.9) -> np.ndarray:
    """使用百分位方法检测离群值"""
    lower_bound = np.percentile(values, lower_pct)
    upper_bound = np.percentile(values, upper_pct)

    outliers = (values < lower_bound) | (values > upper_bound)
    return outliers


def analyze_parameter_distribution(name: str, tensor: torch.Tensor) -> Dict:
    """分析单个参数的分布和离群值"""
    values = tensor.flatten().numpy()

    # 基本统计
    stats = {
        'name': name,
        'shape': str(tensor.shape),
        'total_params': len(values),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'q1': float(np.percentile(values, 25)),
        'q3': float(np.percentile(values, 75)),
        'p01': float(np.percentile(values, 0.1)),
        'p99_9': float(np.percentile(values, 99.9)),
    }

    # IQR离群值检测
    outliers_iqr, lower_iqr, upper_iqr = detect_outliers_iqr(values, factor=1.5)
    stats['outliers_iqr_count'] = int(np.sum(outliers_iqr))
    stats['outliers_iqr_ratio'] = float(np.sum(outliers_iqr) / len(values))
    stats['iqr_lower_bound'] = float(lower_iqr)
    stats['iqr_upper_bound'] = float(upper_iqr)

    # Z-score离群值检测
    outliers_zscore = detect_outliers_zscore(values, threshold=3.0)
    stats['outliers_zscore_count'] = int(np.sum(outliers_zscore))
    stats['outliers_zscore_ratio'] = float(np.sum(outliers_zscore) / len(values))

    # 百分位离群值检测
    outliers_pct = detect_outliers_percentile(values, lower_pct=0.1, upper_pct=99.9)
    stats['outliers_pct_count'] = int(np.sum(outliers_pct))
    stats['outliers_pct_ratio'] = float(np.sum(outliers_pct) / len(values))

    # 计算量化范围的影响
    # 如果包含离群值
    range_with_outliers = stats['max'] - stats['min']
    # 如果去除离群值（使用IQR方法）
    range_without_outliers = upper_iqr - lower_iqr
    stats['range_with_outliers'] = float(range_with_outliers)
    stats['range_without_outliers'] = float(range_without_outliers)
    stats['range_reduction_ratio'] = float(range_without_outliers / range_with_outliers) if range_with_outliers > 0 else 1.0

    # 计算去除离群值后的统计
    inliers = values[~outliers_iqr]
    if len(inliers) > 0:
        stats['inliers_mean'] = float(np.mean(inliers))
        stats['inliers_std'] = float(np.std(inliers))
        stats['inliers_min'] = float(np.min(inliers))
        stats['inliers_max'] = float(np.max(inliers))
    else:
        stats['inliers_mean'] = stats['mean']
        stats['inliers_std'] = stats['std']
        stats['inliers_min'] = stats['min']
        stats['inliers_max'] = stats['max']

    return stats


def categorize_parameters(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """将参数按类型分类"""
    categories = {
        'embedding': {},
        'attention': {},
        'mlp': {},
        'layernorm': {},
        'output': {},
    }

    for key, value in state_dict.items():
        # 只分析weight参数，跳过bias
        if 'bias' in key:
            continue

        if 'wte' in key or 'wpe' in key:
            categories['embedding'][key] = value
        elif 'attn' in key:
            categories['attention'][key] = value
        elif 'mlp' in key:
            categories['mlp'][key] = value
        elif 'ln' in key or 'norm' in key:
            categories['layernorm'][key] = value
        elif 'lm_head' in key or 'output' in key:
            categories['output'][key] = value

    return categories


def analyze_checkpoint(checkpoint_path: str, output_dir: str = './results/outlier_analysis'):
    """分析检查点的离群值"""

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载检查点
    state_dict = load_checkpoint(checkpoint_path)

    # 分类参数
    categorized = categorize_parameters(state_dict)

    # 分析所有weight参数
    all_stats = []

    print("\n" + "="*80)
    print("分析参数分布和离群值...")
    print("="*80)

    for category, params in categorized.items():
        if not params:
            continue

        print(f"\n{category.upper()} 层 ({len(params)} 个参数):")

        for name, tensor in params.items():
            stats = analyze_parameter_distribution(name, tensor)
            stats['category'] = category
            all_stats.append(stats)

            # 打印关键信息
            print(f"\n  {name}:")
            print(f"    形状: {stats['shape']}, 参数量: {stats['total_params']:,}")
            print(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"    IQR离群值: {stats['outliers_iqr_count']:,} ({stats['outliers_iqr_ratio']*100:.2f}%)")
            print(f"    量化范围缩减: {stats['range_reduction_ratio']*100:.1f}% (去除离群值后)")

    # 转换为DataFrame
    df = pd.DataFrame(all_stats)

    # 保存详细统计到CSV
    csv_path = output_path / 'parameter_statistics.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 详细统计已保存到: {csv_path}")

    # 保存汇总统计
    summary_stats = df.groupby('category').agg({
        'total_params': 'sum',
        'outliers_iqr_ratio': 'mean',
        'outliers_zscore_ratio': 'mean',
        'outliers_pct_ratio': 'mean',
        'range_reduction_ratio': 'mean',
    }).reset_index()

    summary_path = output_path / 'category_summary.csv'
    summary_stats.to_csv(summary_path, index=False)
    print(f"✅ 分类汇总已保存到: {summary_path}")

    # 可视化
    visualize_outlier_analysis(df, output_path)

    # 导出离群值详细数据（采样）
    export_outlier_samples(state_dict, categorized, output_path)

    return df


def visualize_outlier_analysis(df: pd.DataFrame, output_path: Path):
    """可视化离群值分析结果"""

    print("\n生成可视化图表...")

    # 1. 离群值比例对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 按类别统计离群值比例
    ax = axes[0, 0]
    category_stats = df.groupby('category')[['outliers_iqr_ratio', 'outliers_zscore_ratio', 'outliers_pct_ratio']].mean()
    category_stats.plot(kind='bar', ax=ax)
    ax.set_title('Average Outlier Ratio by Category', fontsize=14)
    ax.set_ylabel('Outlier Ratio')
    ax.set_xlabel('Category')
    ax.legend(['IQR (1.5)', 'Z-score (>3)', 'Percentile (0.1%-99.9%)'])
    ax.grid(True, alpha=0.3)

    # 量化范围缩减比例
    ax = axes[0, 1]
    df_sorted = df.sort_values('range_reduction_ratio')
    ax.barh(range(len(df_sorted)), df_sorted['range_reduction_ratio'] * 100)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([name.split('.')[-2] + '.' + name.split('.')[-1] for name in df_sorted['name']], fontsize=6)
    ax.set_xlabel('Range Reduction Ratio (%)')
    ax.set_title('Quantization Range Reduction (After Removing Outliers)', fontsize=14)
    ax.axvline(100, color='red', linestyle='--', label='No reduction')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # 参数范围分布
    ax = axes[1, 0]
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        ax.scatter(cat_df['range_with_outliers'], cat_df['range_without_outliers'],
                  label=category, alpha=0.6, s=100)
    ax.plot([0, df['range_with_outliers'].max()], [0, df['range_with_outliers'].max()],
           'r--', label='No change')
    ax.set_xlabel('Range with Outliers')
    ax.set_ylabel('Range without Outliers')
    ax.set_title('Impact of Outlier Removal on Parameter Range', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 离群值数量分布
    ax = axes[1, 1]
    df['outliers_iqr_pct'] = df['outliers_iqr_ratio'] * 100
    df_plot = df.sort_values('outliers_iqr_pct', ascending=False).head(20)
    ax.barh(range(len(df_plot)), df_plot['outliers_iqr_pct'])
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels([name.split('.')[-2] + '.' + name.split('.')[-1] for name in df_plot['name']], fontsize=8)
    ax.set_xlabel('Outlier Percentage (%)')
    ax.set_title('Top 20 Parameters by Outlier Ratio (IQR Method)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    viz_path = output_path / 'outlier_analysis.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化图表已保存到: {viz_path}")
    plt.close()

    # 2. 详细的分布对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    categories = df['category'].unique()
    for idx, category in enumerate(categories):
        if idx >= 6:
            break

        cat_df = df[df['category'] == category]

        ax = axes[idx]

        # 绘制范围缩减
        x = range(len(cat_df))
        width = 0.35

        ax.bar([i - width/2 for i in x], cat_df['range_with_outliers'],
              width, label='With outliers', alpha=0.7)
        ax.bar([i + width/2 for i in x], cat_df['range_without_outliers'],
              width, label='Without outliers', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([name.split('.')[-1] for name in cat_df['name']],
                          rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Parameter Range')
        ax.set_title(f'{category.upper()} - Range Comparison', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # 隐藏多余的子图
    for idx in range(len(categories), 6):
        axes[idx].axis('off')

    plt.tight_layout()
    range_path = output_path / 'range_comparison.png'
    plt.savefig(range_path, dpi=300, bbox_inches='tight')
    print(f"✅ 范围对比图已保存到: {range_path}")
    plt.close()


def export_outlier_samples(state_dict: Dict[str, torch.Tensor],
                          categorized: Dict[str, Dict[str, torch.Tensor]],
                          output_path: Path):
    """导出离群值样本数据"""

    print("\n导出离群值样本数据...")

    # 选择几个代表性的参数导出详细数据
    sample_params = []

    # 从每个类别选择一个参数
    for category, params in categorized.items():
        if params:
            # 选择第一个参数
            first_key = list(params.keys())[0]
            sample_params.append((category, first_key, params[first_key]))

    for category, name, tensor in sample_params:
        values = tensor.flatten().numpy()

        # 检测离群值
        outliers_iqr, lower_iqr, upper_iqr = detect_outliers_iqr(values, factor=1.5)

        # 创建DataFrame
        df = pd.DataFrame({
            'value': values,
            'is_outlier_iqr': outliers_iqr,
            'abs_value': np.abs(values),
        })

        # 排序并采样（保存前1000个最大值和最小值）
        df_sorted = df.sort_values('value')
        df_sample = pd.concat([
            df_sorted.head(1000),  # 最小的1000个
            df_sorted.tail(1000),  # 最大的1000个
        ])

        # 保存
        safe_name = name.replace('.', '_').replace('/', '_')
        sample_path = output_path / f'sample_{category}_{safe_name}.csv'
        df_sample.to_csv(sample_path, index=False)
        print(f"  ✅ {category}/{name}: {len(df_sample)} 个样本已保存")

    print(f"\n✅ 所有样本数据已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='分析检查点参数的离群值分布')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/lihongliang/fangzl/ckpt-compress/checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt',
        help='检查点路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/outlier_analysis',
        help='输出目录'
    )

    args = parser.parse_args()

    df = analyze_checkpoint(args.checkpoint, args.output_dir)

    # 打印汇总
    print("\n" + "="*80)
    print("汇总统计")
    print("="*80)

    print("\n按类别统计:")
    summary = df.groupby('category').agg({
        'total_params': 'sum',
        'outliers_iqr_ratio': 'mean',
        'range_reduction_ratio': 'mean',
    })
    print(summary)

    print("\n关键发现:")
    print(f"  - 平均离群值比例 (IQR): {df['outliers_iqr_ratio'].mean()*100:.2f}%")
    print(f"  - 平均量化范围缩减: {(1-df['range_reduction_ratio'].mean())*100:.1f}%")
    print(f"  - 最大范围缩减: {(1-df['range_reduction_ratio'].min())*100:.1f}%")

    print("\n💡 建议:")
    print("  1. 使用IQR方法剪枝离群值可以显著缩小量化范围")
    print("  2. 不同层的离群值比例差异较大，建议分层处理")
    print("  3. 查看CSV文件了解每个参数的详细统计")


if __name__ == '__main__':
    main()
