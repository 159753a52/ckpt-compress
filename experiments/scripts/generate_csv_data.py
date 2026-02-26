"""
生成 GPT-2 重要性得分的 CSV 数据文件，用于诊断可视化问题。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import numpy as np
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ckpt_compress.methods.adam_prune.importance import compute_importance_scores
from ckpt_compress.models.gpt2 import get_gpt2_small


def get_layer_type(name: str) -> str:
    name_lower = name.lower()
    if 'wte' in name_lower or 'wpe' in name_lower:
        return 'embedding'
    elif 'attn' in name_lower:
        if 'c_attn' in name_lower:
            return 'attention_qkv'
        elif 'c_proj' in name_lower:
            return 'attention_proj'
        else:
            return 'attention_other'
    elif 'mlp' in name_lower:
        if 'c_fc' in name_lower:
            return 'mlp_fc'
        elif 'c_proj' in name_lower:
            return 'mlp_proj'
        else:
            return 'mlp_other'
    elif 'ln' in name_lower:
        return 'layernorm'
    else:
        return 'other'


def get_block_number(name: str) -> int:
    match = re.search(r'\.h\.(\d+)\.', name)
    if match:
        return int(match.group(1))
    return -1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = "./results/gpt2_data"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("生成 GPT-2 重要性得分 CSV 数据")
    print("=" * 70)

    # 创建和训练模型
    print("\n[1] 创建 GPT-2 模型并训练...")
    model = get_gpt2_small(pretrained=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for step in range(100):
        input_ids = torch.randint(0, 50257, (4, 64), device=device)
        optimizer.zero_grad()
        logits = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        loss.backward()
        optimizer.step()

    # 提取重要性得分
    print("\n[2] 提取重要性得分...")
    input_ids = torch.randint(0, 50257, (4, 64), device=device)
    optimizer.zero_grad()
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    loss.backward()

    weights = {}
    gradients = {}
    exp_avg_sq = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            weights[name] = param.data.clone().cpu()
            gradients[name] = param.grad.clone().cpu()
            if param in optimizer.state and 'exp_avg_sq' in optimizer.state[param]:
                exp_avg_sq[name] = optimizer.state[param]['exp_avg_sq'].clone().cpu()

    importance_scores = compute_importance_scores(weights, gradients, exp_avg_sq)

    # 生成汇总统计 CSV
    print("\n[3] 生成汇总统计...")
    summary_path = os.path.join(save_dir, "layer_summary.csv")
    with open(summary_path, 'w') as f:
        f.write("layer_name,layer_type,block_num,n_params,min,max,mean,median,std,skewness\n")

        from scipy import stats
        for name, scores in sorted(importance_scores.items()):
            scores_np = scores.flatten().numpy()
            layer_type = get_layer_type(name)
            block_num = get_block_number(name)

            f.write(f"{name},{layer_type},{block_num},{len(scores_np)},"
                   f"{scores_np.min():.15e},{scores_np.max():.15e},"
                   f"{scores_np.mean():.15e},{np.median(scores_np):.15e},"
                   f"{scores_np.std():.15e},{stats.skew(scores_np):.4f}\n")

    print(f"    保存到: {summary_path}")

    # 为一个示例层生成详细数据（采样）
    print("\n[4] 生成示例层的详细数据...")

    # 选择 mlp_proj 的第一个 block
    example_layer = None
    for name in importance_scores.keys():
        if 'mlp' in name.lower() and 'c_proj' in name.lower() and '.h.0.' in name:
            example_layer = name
            break

    if example_layer:
        scores_np = importance_scores[example_layer].flatten().numpy()

        # 保存完整数据（采样以减小文件大小）
        sample_size = min(100000, len(scores_np))
        sample_indices = np.random.choice(len(scores_np), sample_size, replace=False)
        sample_scores = scores_np[sample_indices]

        detail_path = os.path.join(save_dir, "example_layer_data.csv")
        with open(detail_path, 'w') as f:
            f.write("importance_score\n")
            for score in sample_scores:
                f.write(f"{score:.15e}\n")

        print(f"    示例层: {example_layer}")
        print(f"    采样数量: {sample_size}")
        print(f"    保存到: {detail_path}")

        # 打印数据分布统计
        print(f"\n    数据统计:")
        print(f"        总数: {len(scores_np):,}")
        print(f"        最小值: {scores_np.min():.15e}")
        print(f"        最大值: {scores_np.max():.15e}")
        print(f"        均值: {scores_np.mean():.15e}")
        print(f"        中位数: {np.median(scores_np):.15e}")
        print(f"        标准差: {scores_np.std():.15e}")

        # 分位数
        print(f"\n    分位数:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(scores_np, p)
            print(f"        {p}%: {val:.15e}")

        # 直方图统计
        print(f"\n    直方图统计 (100 bins, 对数空间):")
        scores_positive = scores_np[scores_np > 0]
        log_min = np.log10(scores_positive.min())
        log_max = np.log10(scores_positive.max())
        bins = np.logspace(log_min, log_max, 101)
        counts, bin_edges = np.histogram(scores_positive, bins=bins)

        print(f"        正值数量: {len(scores_positive):,}")
        print(f"        零值数量: {len(scores_np) - len(scores_positive):,}")
        print(f"        Bin 范围: [{10**log_min:.2e}, {10**log_max:.2e}]")
        print(f"        最大 bin count: {counts.max():,}")
        print(f"        最小 bin count: {counts.min():,}")
        print(f"        平均 bin count: {counts.mean():.1f}")

        # 保存直方图数据
        hist_path = os.path.join(save_dir, "example_histogram.csv")
        with open(hist_path, 'w') as f:
            f.write("bin_left,bin_right,count\n")
            for i in range(len(counts)):
                f.write(f"{bin_edges[i]:.15e},{bin_edges[i+1]:.15e},{counts[i]}\n")
        print(f"\n    直方图数据保存到: {hist_path}")

    print("\n" + "=" * 70)
    print("数据生成完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
