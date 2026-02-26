"""
使用 HVP (Hessian-Vector Product) 绝对值重要性计算方式为每个 block 生成分布图。

运行命令:
    python experiments/scripts/analyze_all_blocks_importance_hvp_abs.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --num_batches 50 \
        --batch_size 2 \
        --seq_length 256 \
        --device cuda \
        --output_dir results/blocks_importance_hvp_abs

重要性公式: d_i = |g_i * θ_i - 0.5 * θ_i * (H * θ)_i|

相比原始 HVP 方法：
- 使用绝对值，避免正负值相互抵消
- 分布更宽，更容易解释
- 仍然考虑 Hessian 的非对角元素（参数间相互作用）
- 需要禁用 scaled_dot_product_attention 以支持二阶导数
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
from src.ckpt_compress.methods.adam_prune.importance import compute_importance_scores_hvp_abs


def cache_data_batches(data_loader, num_batches, device):
    """缓存数据批次到内存/显存。"""
    print(f"缓存 {num_batches} 个数据批次...")
    batches = []
    data_iter = iter(data_loader)

    for _ in tqdm(range(num_batches)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        # 将数据移到设备上
        batches.append({
            'input_ids': batch['input_ids'].to(device),
            'labels': batch['labels'].to(device)
        })

    return batches


def compute_importance_scores_hvp_abs_for_analysis(
    model,
    data_batches,
    device,
    num_batches=5
):
    """
    使用 HVP 计算参数重要性得分（绝对值版本）。

    公式: d_i = |g_i * θ_i - 0.5 * θ_i * (H * θ)_i|
    """
    model.eval()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # 定义损失函数
    def loss_fn(model, batch):
        input_ids = batch['input_ids']
        labels = batch['labels']

        logits = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss

    # 计算重要性得分
    print("\n计算 HVP 重要性得分（绝对值版本）...")
    scores = compute_importance_scores_hvp_abs(
        model=model,
        loss_fn=loss_fn,
        data_batches=data_batches,
        num_batches=num_batches
    )

    # 移到 CPU
    scores = {name: tensor.cpu() for name, tensor in scores.items()}

    return scores


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


def plot_block_distributions(block_name, block_params, output_path):
    """绘制单个 block 的参数分布图。"""
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
        scores_filtered = scores[scores < np.percentile(scores, 99)]

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
        ax.set_xlabel('Importance Score (HVP Absolute)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{display_name}\n(n={len(scores):,}, std={std_score:.2e})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')

    # 设置总标题
    fig.suptitle(f'{block_name.replace("_", " ").title()} - HVP Absolute Importance Distribution',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {block_name}: {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='使用 HVP 绝对值重要性为每个 block 生成分布图')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt',
                        help='检查点路径')
    parser.add_argument('--num_batches', type=int, default=50,
                        help='用于 HVP 计算的批次数量')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小（建议较小以节省显存）')
    parser.add_argument('--seq_length', type=int, default=256,
                        help='序列长度（建议较小以节省显存）')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备')
    parser.add_argument('--output_dir', type=str,
                        default='results/blocks_importance_hvp_abs',
                        help='输出目录')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = 'cpu'

    print("=" * 80)
    print("GPT-2 所有 Blocks 的参数重要性分布分析（HVP 绝对值版本）")
    print("=" * 80)
    print(f"检查点: {args.checkpoint}")
    print(f"HVP 批次数: {args.num_batches}")
    print(f"批次大小: {args.batch_size}")
    print(f"序列长度: {args.seq_length}")
    print(f"重要性公式: d_i = |g_i * θ_i - 0.5 * θ_i * (H * θ)_i|")
    print(f"GPT-2 Small: 12 个 Transformer Blocks")
    print("=" * 80)
    print("\n⚠️  警告: HVP 计算需要大量显存（约 3-5 GB）")
    print("   建议使用较小的 batch_size 和 seq_length")
    print("=" * 80)

    # 加载模型
    print("\n加载模型...")
    model = get_gpt2_small(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 模型已加载 (步数: {checkpoint['step']})")

    # 禁用 scaled_dot_product_attention 以支持二阶导数
    print("⚠️  禁用 scaled_dot_product_attention 以支持 HVP 计算...")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # 加载数据
    print("\n加载数据...")
    data_loader = get_wikitext103_dataloader(
        split='train',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0
    )
    print(f"✓ 数据已加载")

    # 缓存数据批次
    data_batches = cache_data_batches(data_loader, args.num_batches, device)

    # 计算重要性得分（HVP 绝对值版本）
    print("\n" + "=" * 80)
    print("开始 HVP 重要性计算（绝对值版本）...")
    print("=" * 80)
    scores = compute_importance_scores_hvp_abs_for_analysis(
        model=model,
        data_batches=data_batches,
        device=device,
        num_batches=args.num_batches
    )
    print(f"✓ 已计算重要性得分")

    # 按 block 分组
    print("\n按 block 分组参数...")
    grouped_params = group_params_by_block(scores)
    print(f"✓ 分为 {len(grouped_params)} 组")

    # 为每个 block 生成图表
    blocks_to_plot = sorted(grouped_params.keys())
    print(f"\n生成 {len(blocks_to_plot)} 个图表...")
    for block_name in blocks_to_plot:
        block_params = grouped_params[block_name]
        output_path = output_dir / f'{block_name}_importance_hvp_abs.png'
        plot_block_distributions(block_name, block_params, output_path)

    print("\n" + "=" * 80)
    print("分析完成！")
    print(f"结果保存在: {output_dir}")
    print(f"生成了 {len(blocks_to_plot)} 个 PNG 文件")
    print("=" * 80)


if __name__ == '__main__':
    main()
