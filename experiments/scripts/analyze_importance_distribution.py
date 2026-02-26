"""
计算并可视化 GPT-2 模型不同参数组的重要性分布。

使用 AdamPrune 的重要性得分公式：
    s_i = -g_i * θ_i + α * v_i * θ_i²

其中：
    - g_i: 梯度
    - θ_i: 权重
    - v_i: Adam 二阶矩 (exp_avg_sq)
    - α: 二阶项权重系数 (默认 0.5)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from tqdm import tqdm
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader


def extract_layer_type(param_name: str) -> str:
    """
    提取参数的层类型。

    参数:
        param_name: 参数名称

    返回:
        层类型字符串
    """
    if 'wte' in param_name:
        return 'embedding.token'
    elif 'wpe' in param_name:
        return 'embedding.position'
    elif 'ln_f' in param_name:
        return 'layernorm.final'
    elif 'ln_1' in param_name or 'ln_2' in param_name:
        return 'layernorm.layer'
    elif 'c_attn' in param_name:
        return 'attention.qkv'
    elif 'c_proj' in param_name and 'attn' in param_name:
        return 'attention.output'
    elif 'c_fc' in param_name:
        return 'mlp.fc1'
    elif 'c_proj' in param_name and 'mlp' in param_name:
        return 'mlp.fc2'
    elif 'lm_head' in param_name:
        return 'lm_head'
    else:
        return 'other'


def collect_gradients_and_momentum(
    model,
    data_loader,
    device,
    num_steps=100,
    alpha=0.5
):
    """
    累积多个步骤的梯度和 Adam 二阶矩。

    参数:
        model: GPT-2 模型
        data_loader: 数据加载器
        device: 设备
        num_steps: 累积的步数
        alpha: 二阶项权重系数

    返回:
        weights: 当前权重
        accumulated_gradients: 累积的梯度
        accumulated_exp_avg_sq: 累积的二阶矩
    """
    model.train()
    model = model.to(device)

    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # 累积梯度和二阶矩
    accumulated_gradients = defaultdict(lambda: 0)
    accumulated_exp_avg_sq = defaultdict(lambda: 0)

    print(f"累积 {num_steps} 步的梯度和动量...")

    data_iter = iter(data_loader)

    for step in tqdm(range(num_steps)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        # 前向传播
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids)

        # 计算损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # 反向传播
        loss.backward()

        # 累积梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                accumulated_gradients[name] = accumulated_gradients[name] + param.grad.detach().cpu()

        # 优化器步骤
        optimizer.step()

        # 累积 exp_avg_sq
        for name, param in model.named_parameters():
            if name in optimizer.state[param]:
                exp_avg_sq = optimizer.state[param]['exp_avg_sq']
                accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] + exp_avg_sq.detach().cpu()

    # 平均
    for name in accumulated_gradients:
        accumulated_gradients[name] = accumulated_gradients[name] / num_steps
        accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] / num_steps

    # 获取当前权重
    weights = {name: param.detach().cpu() for name, param in model.named_parameters()}

    return weights, dict(accumulated_gradients), dict(accumulated_exp_avg_sq)


def compute_importance_scores(
    weights,
    gradients,
    exp_avg_sq,
    alpha=0.5
):
    """
    计算重要性得分。

    公式: s_i = -g_i * θ_i + α * v_i * θ_i²
    """
    scores = {}

    for name, weight in weights.items():
        if name not in exp_avg_sq:
            continue

        grad = gradients.get(name, torch.zeros_like(weight))
        v = exp_avg_sq[name]

        # s_i = -g_i * θ_i + α * v_i * θ_i²
        first_order = -grad * weight
        second_order = alpha * v * weight ** 2

        scores[name] = first_order + second_order

    return scores


def group_scores_by_layer_type(scores):
    """
    按层类型分组重要性得分。
    """
    grouped = defaultdict(list)

    for name, score_tensor in scores.items():
        layer_type = extract_layer_type(name)
        # 只保留非 bias 参数
        if 'bias' not in name:
            grouped[layer_type].extend(score_tensor.flatten().numpy())

    return dict(grouped)


def plot_importance_distributions(grouped_scores, output_path):
    """
    绘制不同参数组的重要性分布图。
    """
    # 设置绘图风格
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 10

    # 创建子图
    layer_types = sorted(grouped_scores.keys())
    n_types = len(layer_types)
    n_cols = 3
    n_rows = (n_types + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten() if n_types > 1 else [axes]

    for idx, layer_type in enumerate(layer_types):
        ax = axes[idx]
        scores = np.array(grouped_scores[layer_type])

        # 过滤极端值（用于更好的可视化）
        scores_filtered = scores[np.abs(scores) < np.percentile(np.abs(scores), 99)]

        # 绘制直方图
        ax.hist(scores_filtered, bins=100, alpha=0.7, color='steelblue', edgecolor='black')

        # 添加统计信息
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores)

        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2e}')
        ax.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.2e}')

        ax.set_xlabel('Importance Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{layer_type}\n(n={len(scores):,}, std={std_score:.2e})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_types, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 分布图已保存: {output_path}")
    plt.close()


def plot_importance_boxplot(grouped_scores, output_path):
    """
    绘制箱线图对比不同参数组的重要性分布。
    """
    plt.figure(figsize=(14, 8))

    # 准备数据
    data = []
    labels = []

    for layer_type in sorted(grouped_scores.keys()):
        scores = np.array(grouped_scores[layer_type])
        # 过滤极端值
        scores_filtered = scores[np.abs(scores) < np.percentile(np.abs(scores), 99)]
        data.append(scores_filtered)
        labels.append(f'{layer_type}\n(n={len(scores):,})')

    # 绘制箱线图
    bp = plt.boxplot(data, labels=labels, patch_artist=True, showfliers=False)

    # 美化
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    plt.xlabel('Layer Type', fontsize=12, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title('Importance Score Distribution by Layer Type', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 箱线图已保存: {output_path}")
    plt.close()


def plot_importance_violin(grouped_scores, output_path):
    """
    绘制小提琴图展示重要性分布。
    """
    plt.figure(figsize=(14, 8))

    # 准备数据
    data = []
    positions = []
    labels = []

    for idx, layer_type in enumerate(sorted(grouped_scores.keys())):
        scores = np.array(grouped_scores[layer_type])
        # 过滤极端值
        scores_filtered = scores[np.abs(scores) < np.percentile(np.abs(scores), 99)]
        data.append(scores_filtered)
        positions.append(idx + 1)
        labels.append(f'{layer_type}\n(n={len(scores):,})')

    # 绘制小提琴图
    parts = plt.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    # 美化
    for pc in parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)

    plt.xticks(positions, labels, rotation=45, ha='right')
    plt.xlabel('Layer Type', fontsize=12, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title('Importance Score Distribution by Layer Type (Violin Plot)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 小提琴图已保存: {output_path}")
    plt.close()


def print_statistics(grouped_scores):
    """
    打印统计信息。
    """
    print("\n" + "=" * 80)
    print("重要性得分统计")
    print("=" * 80)

    for layer_type in sorted(grouped_scores.keys()):
        scores = np.array(grouped_scores[layer_type])

        print(f"\n{layer_type}:")
        print(f"  参数数量: {len(scores):,}")
        print(f"  均值: {np.mean(scores):.6e}")
        print(f"  中位数: {np.median(scores):.6e}")
        print(f"  标准差: {np.std(scores):.6e}")
        print(f"  最小值: {np.min(scores):.6e}")
        print(f"  最大值: {np.max(scores):.6e}")
        print(f"  25% 分位数: {np.percentile(scores, 25):.6e}")
        print(f"  75% 分位数: {np.percentile(scores, 75):.6e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='计算并可视化 GPT-2 参数重要性分布')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt',
                        help='检查点路径')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='累积的训练步数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='序列长度')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='二阶项权重系数')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备')
    parser.add_argument('--output_dir', type=str,
                        default='results/importance_distribution',
                        help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = 'cpu'

    print("=" * 80)
    print("GPT-2 参数重要性分布分析")
    print("=" * 80)
    print(f"检查点: {args.checkpoint}")
    print(f"累积步数: {args.num_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"序列长度: {args.seq_length}")
    print(f"Alpha: {args.alpha}")
    print(f"设备: {device}")
    print("=" * 80)

    # 加载模型
    print("\n加载模型...")
    model = get_gpt2_small(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 模型已加载 (步数: {checkpoint['step']})")

    # 加载数据
    print("\n加载数据...")
    data_loader = get_wikitext103_dataloader(
        split='train',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0
    )
    print(f"✓ 数据已加载 (批次数: {len(data_loader)})")

    # 收集梯度和动量
    weights, gradients, exp_avg_sq = collect_gradients_and_momentum(
        model=model,
        data_loader=data_loader,
        device=device,
        num_steps=args.num_steps,
        alpha=args.alpha
    )

    # 计算重要性得分
    print("\n计算重要性得分...")
    scores = compute_importance_scores(
        weights=weights,
        gradients=gradients,
        exp_avg_sq=exp_avg_sq,
        alpha=args.alpha
    )
    print(f"✓ 已计算 {len(scores)} 个参数的重要性得分")

    # 按层类型分组
    print("\n按层类型分组...")
    grouped_scores = group_scores_by_layer_type(scores)
    print(f"✓ 分为 {len(grouped_scores)} 个参数组")

    # 打印统计信息
    print_statistics(grouped_scores)

    # 绘制分布图
    print("\n绘制可视化图表...")
    plot_importance_distributions(
        grouped_scores,
        output_dir / 'importance_distributions.png'
    )
    plot_importance_boxplot(
        grouped_scores,
        output_dir / 'importance_boxplot.png'
    )
    plot_importance_violin(
        grouped_scores,
        output_dir / 'importance_violin.png'
    )

    print("\n" + "=" * 80)
    print("分析完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
