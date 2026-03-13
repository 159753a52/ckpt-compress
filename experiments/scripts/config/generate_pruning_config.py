#!/usr/bin/env python3
"""
从无损剪枝分析结果生成目标剪枝率配置

策略:
    根据归一化比例分配剪枝额度：
    1. global_ratio = normalized_prunable_ratio_percent × (目标 / total_normalized)
    2. prune_ratio = global_ratio × total_model_params / param_group_params

    例如：目标 10%
    - normalized 总和 = 51.52%
    - 系数 = 10 / 51.52 = 0.1941
    - attn_qkv (block 0): global_ratio = 0.729 × 0.1941 = 0.1415%
    - prune_ratio = 0.1415% × 84,953,088 / 1,769,472 = 0.0679
"""

import pandas as pd
import json
from pathlib import Path
import argparse


def generate_config(df, target_ratio=0.10):
    """
    生成剪枝配置

    Args:
        df: 包含 normalized_prunable_ratio_percent 的 DataFrame
        target_ratio: 目标全局剪枝率

    Returns:
        config: 剪枝配置字典
    """
    total_params = df['total_params'].sum()

    # 计算总归一化比例
    total_normalized = df['normalized_prunable_ratio_percent'].sum()

    # 计算系数
    coefficient = (target_ratio * 100) / total_normalized

    print(f"剪枝配置生成:")
    print(f"  总参数数: {total_params:,}")
    print(f"  目标剪枝率: {target_ratio*100:.1f}%")
    print(f"  归一化比例总和: {total_normalized:.2f}%")
    print(f"  计算系数: {coefficient:.4f}")

    # 初始化配置
    config = {"blocks": {}}

    # 分配剪枝率
    for _, row in df.iterrows():
        block_id = int(row['block_id'])
        layer = row['layer']
        normalized_ratio = row['normalized_prunable_ratio_percent']
        group_params = row['total_params']

        # 步骤1: 计算占模型总参数的百分比
        global_ratio = (normalized_ratio / 100) * coefficient

        # 步骤2: 转换为参数组内部的剪枝比例
        actual_sparsity = global_ratio * total_params / group_params

        bid_str = str(block_id)
        if bid_str not in config["blocks"]:
            config["blocks"][bid_str] = {}

        config["blocks"][bid_str][layer] = round(actual_sparsity, 4)

    # 验证
    actual_pruned = 0
    for _, row in df.iterrows():
        bid_str = str(int(row['block_id']))
        layer = row['layer']
        if bid_str in config["blocks"] and layer in config["blocks"][bid_str]:
            actual_pruned += config["blocks"][bid_str][layer] * row['total_params']

    actual_ratio = actual_pruned / total_params

    print(f"  验证:")
    print(f"    实际剪枝: {actual_pruned:,} ({actual_ratio*100:.2f}%)")

    return config, actual_ratio


def print_summary(config):
    """打印配置摘要"""
    print(f"\n各参数组剪枝率统计:")

    for layer_type in ['attn_qkv', 'attn_proj', 'mlp_fc', 'mlp_proj', 'ln_1', 'ln_2']:
        sparsities = []
        for block_id in range(12):
            bid_str = str(block_id)
            if bid_str in config["blocks"] and layer_type in config["blocks"][bid_str]:
                sparsities.append(config["blocks"][bid_str][layer_type])

        if sparsities:
            print(f"  {layer_type:<12}: min={min(sparsities):.4f}, max={max(sparsities):.4f}, mean={sum(sparsities)/len(sparsities):.4f}")


def save_config(config, output_path):
    """保存配置到 JSON 文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ 配置已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='从无损剪枝分析生成目标剪枝率配置')
    parser.add_argument('--input', type=str,
                        default='results/fine_grained_pruning_analysis/normalized_prunable_ratios.csv',
                        help='输入 CSV 文件路径')
    parser.add_argument('--output', type=str,
                        default='results/pruning_configs/lossless_based_10percent.json',
                        help='输出 JSON 文件路径')
    parser.add_argument('--target', type=float, default=0.10,
                        help='目标剪枝率 (默认: 0.10)')

    args = parser.parse_args()

    # 读取数据
    df = pd.read_csv(args.input)

    # 生成配置
    config, _ = generate_config(df, args.target)

    # 打印摘要
    print_summary(config)

    # 保存配置
    save_config(config, args.output)


if __name__ == '__main__':
    main()
