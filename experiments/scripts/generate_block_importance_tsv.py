"""
生成GPT-2各Block各层重要性得分TSV文件。

基于现有CSV文件计算总重要性和多种归一化得分。
"""

import os
import csv
from collections import defaultdict


def main():
    # 输入输出路径
    input_csv = "results/gpt2_data/layer_summary.csv"
    output_dir = "results/gpt2_block_analysis"
    output_tsv = os.path.join(output_dir, "block_layer_importance.tsv")

    os.makedirs(output_dir, exist_ok=True)

    # 读取CSV数据
    rows = []
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'layer_name': row['layer_name'],
                'layer_type': row['layer_type'],
                'block_num': int(row['block_num']),
                'n_params': int(row['n_params']),
                'mean': float(row['mean']),
            })

    # 计算 total_importance
    for row in rows:
        row['total_importance'] = row['n_params'] * row['mean']

    # 计算全局总重要性
    global_total = sum(r['total_importance'] for r in rows)

    # 按 layer_type 分组计算总重要性
    type_totals = defaultdict(float)
    for row in rows:
        type_totals[row['layer_type']] += row['total_importance']

    # 按 block_num 分组计算总重要性
    block_totals = defaultdict(float)
    for row in rows:
        block_totals[row['block_num']] += row['total_importance']

    # 计算归一化得分
    for row in rows:
        row['norm_global'] = row['total_importance'] / global_total if global_total > 0 else 0
        row['norm_by_type'] = row['total_importance'] / type_totals[row['layer_type']] if type_totals[row['layer_type']] > 0 else 0
        row['norm_by_block'] = row['total_importance'] / block_totals[row['block_num']] if block_totals[row['block_num']] > 0 else 0

    # 按 block_num 和 layer_type 排序
    rows.sort(key=lambda r: (r['block_num'], r['layer_type'], r['layer_name']))

    # 输出TSV文件
    with open(output_tsv, 'w') as f:
        # 写入表头
        headers = ['block_num', 'layer_type', 'param_name', 'n_params', 'mean_importance',
                   'total_importance', 'norm_global', 'norm_by_type', 'norm_by_block']
        f.write('\t'.join(headers) + '\n')

        # 写入数据
        for row in rows:
            values = [
                str(row['block_num']),
                row['layer_type'],
                row['layer_name'],
                str(row['n_params']),
                f"{row['mean']:.10e}",
                f"{row['total_importance']:.10e}",
                f"{row['norm_global']:.10e}",
                f"{row['norm_by_type']:.10e}",
                f"{row['norm_by_block']:.10e}",
            ]
            f.write('\t'.join(values) + '\n')

    print(f"输出文件: {output_tsv}")
    print(f"总行数: {len(rows)}")
    print(f"全局总重要性: {global_total:.10e}")

    # 验证归一化
    print("\n验证归一化:")
    print(f"  norm_global 之和: {sum(r['norm_global'] for r in rows):.6f} (应为 1.0)")

    for layer_type in sorted(type_totals.keys()):
        type_sum = sum(r['norm_by_type'] for r in rows if r['layer_type'] == layer_type)
        print(f"  {layer_type} norm_by_type 之和: {type_sum:.6f}")

    for block_num in sorted(block_totals.keys()):
        block_sum = sum(r['norm_by_block'] for r in rows if r['block_num'] == block_num)
        print(f"  Block {block_num} norm_by_block 之和: {block_sum:.6f}")


if __name__ == "__main__":
    main()
