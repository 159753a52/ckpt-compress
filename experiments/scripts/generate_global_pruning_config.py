#!/usr/bin/env python3
"""
根据归一化可剪枝比例生成全局剪枝配置

使用方法:
python experiments/scripts/generate_global_pruning_config.py \
    --input results/fine_grained_pruning_analysis/normalized_prunable_ratios.csv \
    --global_sparsity 0.10 \
    --output experiments/configs/pruning_config_global_10percent.json
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict


def generate_global_pruning_config(
    normalized_csv_path: str,
    global_sparsity: float,
    output_path: str,
    safety_margin: float = 0.0,
    min_sparsity: float = 0.0,
) -> Dict:
    """
    根据归一化可剪枝比例生成全局剪枝配置

    Args:
        normalized_csv_path: 归一化可剪枝比例CSV文件路径
        global_sparsity: 目标全局剪枝率 (0-1)
        output_path: 输出JSON配置文件路径
        safety_margin: 安全边际，从max_lossless_sparsity中减去的值 (默认0.0)
        min_sparsity: 最小剪枝率阈值，低于此值设为0 (默认0.0)

    Returns:
        生成的配置字典
    """
    # 读取归一化比例数据
    df = pd.read_csv(normalized_csv_path)

    # 计算归一化比例总和
    total_normalized_ratio = df['normalized_prunable_ratio_percent'].sum()

    print(f"归一化比例总和: {total_normalized_ratio:.4f}%")
    print(f"目标全局剪枝率: {global_sparsity * 100:.2f}%")
    print(f"安全边际: {safety_margin * 100:.2f}%")
    print(f"最小剪枝率阈值: {min_sparsity * 100:.2f}%")
    print()

    # 初始化配置
    config = {"blocks": {}}

    # 统计信息
    total_params = 0
    total_pruned_params = 0
    capped_count = 0
    zeroed_count = 0

    # 计算缩放因子
    # 如果目标全局剪枝率是10%，而最大可能是51.52%，则缩放因子 = 10/51.52 = 0.194
    scale_factor = global_sparsity / (total_normalized_ratio / 100.0)

    print(f"缩放因子: {scale_factor:.6f}")
    print(f"  (即每个参数组的max_lossless_sparsity将乘以此因子)")
    print()

    # 按block和layer生成配置
    for _, row in df.iterrows():
        layer = row['layer']
        block_id = int(row['block_id'])
        total_params_layer = row['total_params']
        max_lossless = row['max_lossless_sparsity']
        normalized_ratio = row['normalized_prunable_ratio_percent']

        # 计算分配的剪枝率
        # 公式: sparsity = max_lossless_sparsity × scale_factor
        # 这样可以保证按比例分配，且总剪枝率接近目标
        allocated_sparsity = max_lossless * scale_factor

        # 应用安全边际
        safe_max_lossless = max(0.0, max_lossless - safety_margin)

        # 确保不超过最大无损剪枝率（考虑安全边际）
        if allocated_sparsity > safe_max_lossless:
            final_sparsity = safe_max_lossless
            capped_count += 1
        else:
            final_sparsity = allocated_sparsity

        # 应用最小剪枝率阈值
        if final_sparsity < min_sparsity:
            final_sparsity = 0.0
            zeroed_count += 1

        # 四舍五入到小数点后4位（保留更高精度）
        final_sparsity = round(final_sparsity, 4)

        # 更新统计
        total_params += total_params_layer
        total_pruned_params += total_params_layer * final_sparsity

        # 添加到配置
        block_key = str(block_id)
        if block_key not in config["blocks"]:
            config["blocks"][block_key] = {}

        config["blocks"][block_key][layer] = final_sparsity

    # 计算实际全局剪枝率
    actual_global_sparsity = total_pruned_params / total_params

    print(f"配置生成完成:")
    print(f"  - 总参数数: {total_params:,}")
    print(f"  - 剪枝参数数: {total_pruned_params:,.0f}")
    print(f"  - 实际全局剪枝率: {actual_global_sparsity * 100:.4f}%")
    print(f"  - 目标全局剪枝率: {global_sparsity * 100:.2f}%")
    print(f"  - 偏差: {(actual_global_sparsity - global_sparsity) * 100:.4f}%")
    print(f"  - 被上限限制的参数组数: {capped_count}")
    print(f"  - 被设为0的参数组数: {zeroed_count}")
    print()

    # 保存配置
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"配置已保存到: {output_path}")

    # 生成详细报告
    report_path = output_path.parent / f"{output_path.stem}_report.txt"
    generate_report(df, config, total_normalized_ratio, global_sparsity,
                   actual_global_sparsity, safety_margin, report_path)

    return config


def generate_report(
    df: pd.DataFrame,
    config: Dict,
    total_normalized_ratio: float,
    target_global_sparsity: float,
    actual_global_sparsity: float,
    safety_margin: float,
    report_path: Path,
):
    """生成详细的配置报告"""

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("全局剪枝配置生成报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"目标全局剪枝率: {target_global_sparsity * 100:.2f}%\n")
        f.write(f"实际全局剪枝率: {actual_global_sparsity * 100:.4f}%\n")
        f.write(f"偏差: {(actual_global_sparsity - target_global_sparsity) * 100:.4f}%\n")
        f.write(f"安全边际: {safety_margin * 100:.2f}%\n")
        f.write(f"归一化比例总和: {total_normalized_ratio:.4f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("各参数组详细配置\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Block':<6} {'Layer':<12} {'Total Params':<15} {'Max Lossless':<13} "
                f"{'Normalized %':<14} {'Allocated %':<13} {'Final %':<10} {'Status':<10}\n")
        f.write("-" * 110 + "\n")

        for _, row in df.iterrows():
            layer = row['layer']
            block_id = int(row['block_id'])
            total_params = row['total_params']
            max_lossless = row['max_lossless_sparsity']
            normalized_ratio = row['normalized_prunable_ratio_percent']

            # 计算分配的剪枝率
            allocated_sparsity = (normalized_ratio / total_normalized_ratio) * target_global_sparsity

            # 获取最终配置的剪枝率
            final_sparsity = config["blocks"][str(block_id)][layer]

            # 判断状态
            if final_sparsity == 0.0:
                status = "ZEROED"
            elif final_sparsity < allocated_sparsity:
                status = "CAPPED"
            else:
                status = "OK"

            f.write(f"{block_id:<6} {layer:<12} {total_params:<15,} {max_lossless * 100:<12.2f}% "
                   f"{normalized_ratio:<13.4f}% {allocated_sparsity * 100:<12.4f}% "
                   f"{final_sparsity * 100:<9.2f}% {status:<10}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("按层类型统计\n")
        f.write("=" * 80 + "\n\n")

        # 按层类型统计
        layer_stats = {}
        for _, row in df.iterrows():
            layer = row['layer']
            block_id = int(row['block_id'])
            final_sparsity = config["blocks"][str(block_id)][layer]

            if layer not in layer_stats:
                layer_stats[layer] = []
            layer_stats[layer].append(final_sparsity)

        f.write(f"{'Layer Type':<12} {'Min %':<10} {'Max %':<10} {'Mean %':<10} {'Std %':<10}\n")
        f.write("-" * 52 + "\n")

        for layer, sparsities in sorted(layer_stats.items()):
            import numpy as np
            min_s = np.min(sparsities) * 100
            max_s = np.max(sparsities) * 100
            mean_s = np.mean(sparsities) * 100
            std_s = np.std(sparsities) * 100

            f.write(f"{layer:<12} {min_s:<9.4f}% {max_s:<9.4f}% {mean_s:<9.4f}% {std_s:<9.4f}%\n")

    print(f"详细报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="根据归一化可剪枝比例生成全局剪枝配置"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="归一化可剪枝比例CSV文件路径"
    )
    parser.add_argument(
        "--global_sparsity",
        type=float,
        required=True,
        help="目标全局剪枝率 (0-1，例如0.10表示10%%)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出JSON配置文件路径"
    )
    parser.add_argument(
        "--safety_margin",
        type=float,
        default=0.0,
        help="安全边际，从max_lossless_sparsity中减去的值 (默认0.0)"
    )
    parser.add_argument(
        "--min_sparsity",
        type=float,
        default=0.0,
        help="最小剪枝率阈值，低于此值设为0 (默认0.0)"
    )

    args = parser.parse_args()

    # 验证参数
    if not (0 <= args.global_sparsity <= 1):
        raise ValueError("global_sparsity 必须在 0-1 之间")

    if not (0 <= args.safety_margin <= 1):
        raise ValueError("safety_margin 必须在 0-1 之间")

    if not (0 <= args.min_sparsity <= 1):
        raise ValueError("min_sparsity 必须在 0-1 之间")

    # 生成配置
    generate_global_pruning_config(
        normalized_csv_path=args.input,
        global_sparsity=args.global_sparsity,
        output_path=args.output,
        safety_margin=args.safety_margin,
        min_sparsity=args.min_sparsity,
    )


if __name__ == "__main__":
    main()
