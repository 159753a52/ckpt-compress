"""
分析单个张量的参数重要性分布。

只分析 embedding 层和第一个 transformer block 的参数。
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader


def collect_gradients_and_momentum(
    model,
    data_loader,
    device,
    num_steps=100,
):
    """累积多个步骤的梯度和 Adam 二阶矩。"""
    model.train()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

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

        for name, param in model.named_parameters():
            if param.grad is not None:
                accumulated_gradients[name] = accumulated_gradients[name] + param.grad.detach().cpu()

        optimizer.step()

        for name, param in model.named_parameters():
            if name in optimizer.state[param]:
                exp_avg_sq = optimizer.state[param]['exp_avg_sq']
                accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] + exp_avg_sq.detach().cpu()

    # 平均
    for name in accumulated_gradients:
        accumulated_gradients[name] = accumulated_gradients[name] / num_steps
        accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] / num_steps

    weights = {name: param.detach().cpu() for name, param in model.named_parameters()}

    return weights, dict(accumulated_gradients), dict(accumulated_exp_avg_sq)


def compute_importance_scores(weights, gradients, exp_avg_sq, alpha=0.5):
    """计算重要性得分: s_i = -g_i * θ_i + α * v_i * θ_i²"""
    scores = {}

    for name, weight in weights.items():
        if name not in exp_avg_sq:
            continue

        grad = gradients.get(name, torch.zeros_like(weight))
        v = exp_avg_sq[name]

        first_order = -grad * weight
        second_order = alpha * v * weight ** 2

        scores[name] = first_order + second_order

    return scores


def filter_target_params(scores):
    """只保留 embedding 层和第一个 block 的参数。"""
    target_params = {}

    for name, score_tensor in scores.items():
        # Embedding 层
        if 'wte.weight' in name or 'wpe.weight' in name:
            target_params[name] = score_tensor
        # 第一个 transformer block
        elif 'transformer.h.0.' in name:
            target_params[name] = score_tensor

    return target_params


def get_param_display_name(param_name):
    """获取参数的显示名称。"""
    if 'wte.weight' in param_name:
        return 'Embedding Token'
    elif 'wpe.weight' in param_name:
        return 'Embedding Position'
    elif 'transformer.h.0.' in param_name:
        # 提取层内的具体参数名
        parts = param_name.split('.')
        if 'ln_1' in param_name:
            param_type = 'weight' if 'weight' in param_name else 'bias'
            return f'Block0 LN1 {param_type}'
        elif 'ln_2' in param_name:
            param_type = 'weight' if 'weight' in param_name else 'bias'
            return f'Block0 LN2 {param_type}'
        elif 'attn.c_attn' in param_name:
            param_type = 'weight' if 'weight' in param_name else 'bias'
            return f'Block0 Attn QKV {param_type}'
        elif 'attn.c_proj' in param_name:
            param_type = 'weight' if 'weight' in param_name else 'bias'
            return f'Block0 Attn Out {param_type}'
        elif 'mlp.c_fc' in param_name:
            param_type = 'weight' if 'weight' in param_name else 'bias'
            return f'Block0 MLP FC1 {param_type}'
        elif 'mlp.c_proj' in param_name:
            param_type = 'weight' if 'weight' in param_name else 'bias'
            return f'Block0 MLP FC2 {param_type}'

    return param_name


def plot_single_tensor_distributions(target_params, output_path):
    """绘制单个张量的重要性分布图。"""
    # 只保留 weight 参数，过滤掉 bias
    weight_params = {name: scores for name, scores in target_params.items() if 'weight' in name}

    n_params = len(weight_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    param_names = sorted(weight_params.keys())

    for idx, param_name in enumerate(param_names):
        ax = axes[idx]
        scores = weight_params[param_name].flatten().numpy()

        # 过滤极端值
        scores_filtered = scores[np.abs(scores) < np.percentile(np.abs(scores), 99)]

        # 绘制直方图
        ax.hist(scores_filtered, bins=100, alpha=0.7, color='steelblue', edgecolor='black')

        # 统计信息
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores)

        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_score:.2e}')
        ax.axvline(median_score, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_score:.2e}')

        display_name = get_param_display_name(param_name)
        ax.set_xlabel('Importance Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{display_name}\n(n={len(scores):,}, std={std_score:.2e})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 分布图已保存: {output_path}")
    plt.close()


def print_statistics(target_params):
    """打印统计信息。"""
    print("\n" + "=" * 80)
    print("单个张量的重要性得分统计")
    print("=" * 80)

    # 只显示 weight 参数
    weight_params = {name: scores for name, scores in target_params.items() if 'weight' in name}

    for param_name in sorted(weight_params.keys()):
        scores = weight_params[param_name].flatten().numpy()
        display_name = get_param_display_name(param_name)

        print(f"\n{display_name}:")
        print(f"  参数名: {param_name}")
        print(f"  参数数量: {len(scores):,}")
        print(f"  形状: {tuple(weight_params[param_name].shape)}")
        print(f"  均值: {np.mean(scores):.6e}")
        print(f"  中位数: {np.median(scores):.6e}")
        print(f"  标准差: {np.std(scores):.6e}")
        print(f"  最小值: {np.min(scores):.6e}")
        print(f"  最大值: {np.max(scores):.6e}")
        print(f"  25% 分位数: {np.percentile(scores, 25):.6e}")
        print(f"  75% 分位数: {np.percentile(scores, 75):.6e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='分析单个张量的参数重要性分布')
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
                        default='results/single_tensor_importance',
                        help='输出目录')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = 'cpu'

    print("=" * 80)
    print("单个张量的参数重要性分布分析")
    print("=" * 80)
    print(f"检查点: {args.checkpoint}")
    print(f"累积步数: {args.num_steps}")
    print(f"分析范围: Embedding 层 + 第一个 Transformer Block")
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
    print(f"✓ 数据已加载")

    # 收集梯度和动量
    weights, gradients, exp_avg_sq = collect_gradients_and_momentum(
        model=model,
        data_loader=data_loader,
        device=device,
        num_steps=args.num_steps,
    )

    # 计算重要性得分
    print("\n计算重要性得分...")
    scores = compute_importance_scores(
        weights=weights,
        gradients=gradients,
        exp_avg_sq=exp_avg_sq,
        alpha=args.alpha
    )
    print(f"✓ 已计算重要性得分")

    # 过滤目标参数
    print("\n过滤目标参数...")
    target_params = filter_target_params(scores)
    print(f"✓ 选择了 {len(target_params)} 个参数张量")

    # 打印统计信息
    print_statistics(target_params)

    # 绘制分布图
    print("\n绘制可视化图表...")
    plot_single_tensor_distributions(
        target_params,
        output_dir / 'single_tensor_distributions.png'
    )

    print("\n" + "=" * 80)
    print("分析完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
