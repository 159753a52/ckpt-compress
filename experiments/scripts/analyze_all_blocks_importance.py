"""
为每个 transformer block 和 embedding 层生成独立的重要性分布图。

每个 block 保存为单独的 PNG 文件。

运行命令:
    python experiments/scripts/analyze_all_blocks_importance.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --num_steps 100 \
        --batch_size 4 \
        --seq_length 512 \
        --alpha 0.5 \
        --device cuda \
        --output_dir results/blocks_importance

重要性公式: s_i = -g_i * θ_i + α * v_i * θ_i² (保留符号)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader


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


def group_params_by_block(scores):
    """按 block 分组参数。"""
    grouped = defaultdict(dict)

    for name, score_tensor in scores.items():
        # Embedding 层
        if 'wte.weight' in name or 'wpe.weight' in name:
            grouped['embedding'][name] = score_tensor
        # Transformer blocks
        elif 'transformer.h.' in name:
            # 提取 block 编号
            parts = name.split('.')
            block_idx = int(parts[2])
            grouped[f'block_{block_idx}'][name] = score_tensor
        # Final layer norm
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
        parts = param_name.split('.')
        block_idx = parts[2]

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


def plot_block_distributions(block_name, block_params, output_path):
    """绘制单个 block 的参数分布图。"""
    # 只保留 weight 参数
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

    # 设置总标题
    fig.suptitle(f'{block_name.replace("_", " ").title()} - Importance Distribution',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {block_name}: {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='为每个 block 生成独立的重要性分布图')
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
                        default='results/blocks_importance',
                        help='输出目录')
    parser.add_argument('--blocks', type=str, default='all',
                        help='要分析的 blocks，例如 "0,1,2" 或 "all"')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = 'cpu'

    print("=" * 80)
    print("GPT-2 所有 Blocks 的参数重要性分布分析")
    print("=" * 80)
    print(f"检查点: {args.checkpoint}")
    print(f"累积步数: {args.num_steps}")
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

    # 计算重要性得分
    print("\n计算重要性得分...")
    scores = compute_importance_scores(
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

    # 确定要绘制的 blocks
    if args.blocks == 'all':
        blocks_to_plot = sorted(grouped_params.keys())
    else:
        block_indices = [int(x.strip()) for x in args.blocks.split(',')]
        blocks_to_plot = ['embedding'] + [f'block_{i}' for i in block_indices]
        blocks_to_plot = [b for b in blocks_to_plot if b in grouped_params]

    # 为每个 block 生成图表
    print(f"\n生成 {len(blocks_to_plot)} 个图表...")
    for block_name in blocks_to_plot:
        block_params = grouped_params[block_name]
        output_path = output_dir / f'{block_name}_importance.png'
        plot_block_distributions(block_name, block_params, output_path)

    print("\n" + "=" * 80)
    print("分析完成！")
    print(f"结果保存在: {output_dir}")
    print(f"生成了 {len(blocks_to_plot)} 个 PNG 文件")
    print("=" * 80)


if __name__ == '__main__':
    main()
