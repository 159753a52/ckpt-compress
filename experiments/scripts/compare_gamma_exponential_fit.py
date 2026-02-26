"""
对比 Gamma 分布和指数分布的拟合效果。

运行命令:
    python experiments/scripts/compare_gamma_exponential_fit.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --num_steps 100 \
        --batch_size 4 \
        --seq_length 512 \
        --alpha 0.5 \
        --device cuda \
        --output_dir results/blocks_importance_gamma_fitted

功能:
- 同时拟合 Gamma 分布和指数分布
- 在直方图上叠加两条拟合曲线
- 对比拟合优度（KS 检验）
- 显示哪个分布拟合更好

Gamma 分布: f(x; k, θ) = (1/(Γ(k)*θ^k)) * x^(k-1) * exp(-x/θ)
指数分布: f(x; λ) = λ * exp(-λ*x)  (Gamma 分布的特例，k=1)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm
from collections import defaultdict
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader
from src.ckpt_compress.methods.adam_prune.importance import compute_importance_scores_abs


def collect_gradients_and_momentum(model, data_loader, device, num_steps=100):
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

    for name in accumulated_gradients:
        accumulated_gradients[name] = accumulated_gradients[name] / num_steps
        accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] / num_steps

    weights = {name: param.detach().cpu() for name, param in model.named_parameters()}

    return weights, dict(accumulated_gradients), dict(accumulated_exp_avg_sq)


def fit_distributions(data):
    """
    同时拟合 Gamma 分布和指数分布。

    返回:
        gamma_params: (k, loc, scale)
        gamma_ks: (statistic, pvalue)
        exp_params: (loc, scale)
        exp_ks: (statistic, pvalue)
    """
    data_positive = data[data > 0]

    if len(data_positive) == 0:
        return None, None, None, None

    # 拟合 Gamma 分布
    gamma_params = stats.gamma.fit(data_positive, floc=0)  # floc=0 固定位置参数为 0
    gamma_ks = stats.kstest(data_positive, 'gamma', args=gamma_params)

    # 拟合指数分布
    exp_params = stats.expon.fit(data_positive, floc=0)
    exp_ks = stats.kstest(data_positive, 'expon', args=exp_params)

    return gamma_params, gamma_ks, exp_params, exp_ks


def group_params_by_block(scores):
    """按 block 分组参数。"""
    grouped = defaultdict(dict)

    for name, score_tensor in scores.items():
        if 'wte.weight' in name or 'wpe.weight' in name:
            grouped['embedding'][name] = score_tensor
        elif 'transformer.h.' in name:
            parts = name.split('.')
            block_idx = int(parts[2])
            grouped[f'block_{block_idx}'][name] = score_tensor
        elif 'ln_f' in name:
            grouped['final_ln'][name] = score_tensor

    return dict(grouped)


def get_param_display_name(param_name):
    """获取参数的显示名称。"""
    if 'wte.weight' in param_name:
        return 'Token Embedding'
    elif 'wpe.weight' in param_name:
        return 'Position Embedding'
    elif 'ln_f.weight' in param_name:
        return 'Final LayerNorm'
    elif 'transformer.h.' in param_name:
        if 'ln_1' in param_name:
            param_type = 'weight' if 'weight' in param_name else 'bias'
            return f'LN1 {param_type}'
        elif 'ln_2' in param_name:
            param_type = 'weight' if 'weight' in param_name else 'bias'
            return f'LN2 {param_type}'
        elif 'attn.c_attn' in param_name:
            param_type = 'weight' if 'weight' in param_name else 'bias'
            return f'Attn QKV {param_type}'
        elif 'attn.c_proj' in param_name:
            param_type = 'weight' if 'weight' in param_name else 'bias'
            return f'Attn Out {param_type}'
        elif 'mlp.c_fc' in param_name:
            param_type = 'weight' if 'weight' in param_name else 'bias'
            return f'MLP FC1 {param_type}'
        elif 'mlp.c_proj' in param_name:
            param_type = 'weight' if 'weight' in param_name else 'bias'
            return f'MLP FC2 {param_type}'

    return param_name


def plot_block_distributions_with_both_fits(block_name, block_params, output_path):
    """绘制单个 block 的参数分布图，并叠加 Gamma 和指数分布拟合曲线。"""
    weight_params = {name: scores for name, scores in block_params.items() if 'weight' in name}

    if not weight_params:
        print(f"  跳过 {block_name}（无 weight 参数）")
        return

    n_params = len(weight_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes

    param_names = sorted(weight_params.keys())

    for idx, param_name in enumerate(param_names):
        ax = axes[idx]
        scores = weight_params[param_name].flatten().numpy()

        # 过滤极端值用于显示
        scores_filtered = scores[scores < np.percentile(scores, 99)]

        # 绘制直方图（归一化为概率密度）
        ax.hist(
            scores_filtered,
            bins=100,
            alpha=0.5,
            color='steelblue',
            edgecolor='black',
            density=True,
            label='Empirical'
        )

        # 拟合两种分布
        gamma_params, gamma_ks, exp_params, exp_ks = fit_distributions(scores)

        if gamma_params is not None:
            x_fit = np.linspace(0, np.max(scores_filtered), 1000)

            # Gamma 分布拟合曲线
            k, loc, scale = gamma_params
            y_gamma = stats.gamma.pdf(x_fit, k, loc, scale)
            ax.plot(x_fit, y_gamma, 'g-', linewidth=2.5,
                   label=f'Gamma (k={k:.2f}, θ={scale:.2e})')

            # 指数分布拟合曲线
            loc_exp, scale_exp = exp_params
            y_exp = stats.expon.pdf(x_fit, loc_exp, scale_exp)
            ax.plot(x_fit, y_exp, 'r--', linewidth=2,
                   label=f'Exponential (λ={1/scale_exp:.2e})')

            # 对比拟合优度
            better_fit = "Gamma" if gamma_ks.pvalue > exp_ks.pvalue else "Exponential"
            better_color = "green" if better_fit == "Gamma" else "red"

            fit_text = f'Gamma: p={gamma_ks.pvalue:.3f}\nExp: p={exp_ks.pvalue:.3f}\nBetter: {better_fit}'
            ax.text(0.98, 0.97, fit_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor=better_color, alpha=0.2),
                   fontsize=8)

        # 统计信息
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores)

        ax.axvline(mean_score, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Mean: {mean_score:.2e}')

        display_name = get_param_display_name(param_name)
        ax.set_xlabel('Importance Score (Absolute)', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.set_title(f'{display_name}\n(n={len(scores):,}, std={std_score:.2e})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')

    # 设置总标题
    fig.suptitle(f'{block_name.replace("_", " ").title()} - Gamma vs Exponential Fit',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {block_name}: {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='对比 Gamma 和指数分布拟合效果')
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
                        default='results/blocks_importance_gamma_fitted',
                        help='输出目录')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = 'cpu'

    print("=" * 80)
    print("GPT-2 参数重要性分布: Gamma vs 指数分布拟合对比")
    print("=" * 80)
    print(f"检查点: {args.checkpoint}")
    print(f"累积步数: {args.num_steps}")
    print(f"重要性公式: d_i = |g_i * θ_i| + α * |v_i * θ_i²|")
    print(f"拟合分布:")
    print(f"  - Gamma: f(x; k, θ) = (1/(Γ(k)*θ^k)) * x^(k-1) * exp(-x/θ)")
    print(f"  - Exponential: f(x; λ) = λ * exp(-λ*x)")
    print(f"GPT-2 Small: 12 个 Transformer Blocks")
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

    # 计算重要性得分（绝对值版本）
    print("\n计算重要性得分（绝对值版本）...")
    scores = compute_importance_scores_abs(
        weights=weights,
        gradients=gradients,
        exp_avg_sq=exp_avg_sq,
        alpha=args.alpha
    )
    print(f"✓ 已计算重要性得分")

    # 按 block 分组
    print("\n按 block 分组参数...")
    grouped_params = group_params_by_block(scores)
    print(f"✓ 分为 {len(grouped_params)} 组")

    # 为每个 block 生成图表（带两种拟合曲线）
    blocks_to_plot = sorted(grouped_params.keys())
    print(f"\n生成 {len(blocks_to_plot)} 个图表（Gamma vs 指数分布）...")
    for block_name in blocks_to_plot:
        block_params = grouped_params[block_name]
        output_path = output_dir / f'{block_name}_gamma_vs_exp.png'
        plot_block_distributions_with_both_fits(block_name, block_params, output_path)

    print("\n" + "=" * 80)
    print("分析完成！")
    print(f"结果保存在: {output_dir}")
    print(f"生成了 {len(blocks_to_plot)} 个 PNG 文件")
    print("\n图例说明:")
    print("  - 绿色实线: Gamma 分布拟合")
    print("  - 红色虚线: 指数分布拟合")
    print("  - 绿色标注: Gamma 拟合更好")
    print("  - 红色标注: 指数拟合更好")
    print("=" * 80)


if __name__ == '__main__':
    main()
