"""
基于 Gamma 分布的自适应剪枝实验（纯一阶重要性）。

运行命令:
    python experiments/scripts/gamma_adaptive_pruning_first_order.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --num_steps 100 \
        --global_prune_ratio 0.1 \
        --eval_batches 10 \
        --batch_size 4 \
        --seq_length 512 \
        --device cuda

实验流程:
1. 计算参数重要性得分（纯一阶：d_i = |g_i * θ_i|）
2. 为每层拟合 Gamma 分布 (k_ℓ, θ_ℓ)
3. 求解全局阈值 τ* 使得全局剪枝率 = 10%
4. 计算各层自适应剪枝率 p_ℓ = P(k_ℓ, τ*/θ_ℓ)
5. 执行剪枝并评估损失差距
6. 生成参数组级别的剪枝比例热力图

理论依据:
- 全局阈值方程: ρ = (1/N) Σ n_ℓ · P(k_ℓ, τ/θ_ℓ)
- 各层剪枝率: p_ℓ = P(k_ℓ, τ*/θ_ℓ)

与标准版本的区别:
- 标准版本: d_i = |g_i * θ_i| + α * |v_i * θ_i²|
- 本版本: d_i = |g_i * θ_i| (纯一阶)

参数组级别粒度:
    Attention:
        - attn_qkv: c_attn.weight [768, 2304] (QKV 投影矩阵)
        - attn_proj: c_proj.weight [768, 768] (输出投影矩阵)
    MLP:
        - mlp_fc: c_fc.weight [768, 3072] (第一层/扩展层)
        - mlp_proj: c_proj.weight [3072, 768] (第二层/投影层)
    LayerNorm:
        - ln_1: ln_1.weight [768] (Attention 前)
        - ln_2: ln_2.weight [768] (MLP 前)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from collections import defaultdict
from scipy import stats
from scipy.optimize import bisect
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader
from src.ckpt_compress.methods.adam_prune.importance import compute_importance_scores_first_order


def collect_gradients_and_momentum(model, data_loader, device, num_steps=100):
    """累积梯度和动量。"""
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
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
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


def fit_gamma_distribution(data):
    """拟合 Gamma 分布，返回 (k, θ)。"""
    data_positive = data[data > 0]
    if len(data_positive) == 0:
        return None, None

    # 拟合 Gamma 分布
    k, loc, scale = stats.gamma.fit(data_positive, floc=0)
    return k, scale  # (shape, scale)


def solve_global_threshold(layer_info, global_prune_ratio):
    """
    求解全局阈值 τ*，使得全局剪枝率 = global_prune_ratio。

    方程: ρ = (1/N) Σ n_ℓ · P(k_ℓ, τ/θ_ℓ)

    参数:
        layer_info: list of dict, 每个包含 {'name', 'n_params', 'k', 'theta', 'scores'}
        global_prune_ratio: 全局剪枝率 (0-1)

    返回:
        tau_star: 全局阈值
    """
    # 计算总参数数
    N = sum(info['n_params'] for info in layer_info)

    def objective(tau):
        """目标函数: 当前阈值下的全局剪枝率 - 目标剪枝率"""
        total_pruned = 0
        for info in layer_info:
            k, theta = info['k'], info['theta']
            if k is None or theta is None:
                continue
            # P(k, τ/θ) = CDF of Gamma(k, θ) at τ
            cdf_value = stats.gamma.cdf(tau, k, scale=theta)
            total_pruned += info['n_params'] * cdf_value

        current_ratio = total_pruned / N
        return current_ratio - global_prune_ratio

    # 找到合适的搜索范围
    all_scores = []
    for info in layer_info:
        if info['scores'] is not None:
            all_scores.extend(info['scores'].flatten().tolist())

    all_scores = np.array(all_scores)
    tau_min = np.percentile(all_scores, 1)
    tau_max = np.percentile(all_scores, 50)

    # 使用二分法求解
    try:
        tau_star = bisect(objective, tau_min, tau_max, xtol=1e-10, maxiter=100)
    except ValueError:
        # 如果二分法失败，尝试更大的范围
        tau_max = np.percentile(all_scores, 90)
        tau_star = bisect(objective, tau_min, tau_max, xtol=1e-10, maxiter=100)

    return tau_star


def compute_layer_prune_ratios(layer_info, tau_star):
    """
    计算各层剪枝率: p_ℓ = P(k_ℓ, τ*/θ_ℓ)

    参数:
        layer_info: list of dict
        tau_star: 全局阈值

    返回:
        更新 layer_info，添加 'prune_ratio' 字段
    """
    for info in layer_info:
        k, theta = info['k'], info['theta']
        if k is None or theta is None:
            info['prune_ratio'] = 0.0
        else:
            # p_ℓ = CDF(τ*)
            info['prune_ratio'] = stats.gamma.cdf(tau_star, k, scale=theta)

    return layer_info


def apply_pruning(model, layer_info, tau_star):
    """
    应用剪枝：将重要性 < τ* 的参数置零。

    返回:
        pruned_model: 剪枝后的模型
        mask_dict: 剪枝掩码字典
    """
    mask_dict = {}

    for info in layer_info:
        name = info['name']
        scores = info['scores']

        if scores is None:
            continue

        # 创建掩码：保留重要性 >= τ* 的参数
        mask = (scores >= tau_star).float()
        mask_dict[name] = mask

        # 应用掩码到模型参数
        param = dict(model.named_parameters())[name]
        param.data.mul_(mask.to(param.device))

    return model, mask_dict


def evaluate_loss(model, data_loader, device, num_batches=10):
    """评估模型在小批量数据上的损失。"""
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    data_iter = iter(data_loader)

    with torch.no_grad():
        for _ in range(num_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            total_loss += loss.item()

    return total_loss / num_batches


def parse_param_group_name(name):
    """
    解析参数名称，返回 (block_id, param_group)

    参数组分类:
        - attn_qkv: transformer.h.{block_id}.attn.c_attn.weight
        - attn_proj: transformer.h.{block_id}.attn.c_proj.weight
        - mlp_fc: transformer.h.{block_id}.mlp.c_fc.weight
        - mlp_proj: transformer.h.{block_id}.mlp.c_proj.weight
        - ln_1: transformer.h.{block_id}.ln_1.weight
        - ln_2: transformer.h.{block_id}.ln_2.weight
        - embedding: wte.weight 或 wpe.weight
    """
    if 'wte.weight' in name or 'wpe.weight' in name:
        return -1, 'embedding'

    # 解析 block id 和参数组
    parts = name.split('.')

    if 'transformer' in parts and 'h' in parts:
        h_idx = parts.index('h')
        block_id = int(parts[h_idx + 1])

        if 'attn' in parts:
            attn_idx = parts.index('attn')
            if 'c_attn' in parts:
                return block_id, 'attn_qkv'
            elif 'c_proj' in parts:
                return block_id, 'attn_proj'
        elif 'mlp' in parts:
            mlp_idx = parts.index('mlp')
            if 'c_fc' in parts:
                return block_id, 'mlp_fc'
            elif 'c_proj' in parts:
                return block_id, 'mlp_proj'
        elif 'ln_1' in parts:
            return block_id, 'ln_1'
        elif 'ln_2' in parts:
            return block_id, 'ln_2'

    return None, None


def aggregate_prune_ratios_by_param_group(layer_info):
    """
    按参数组聚合剪枝比例

    返回: DataFrame with columns [block_id, param_group, avg_prune_ratio, n_params, k, theta]
    """
    param_group_data = []

    for info in layer_info:
        name = info['name']
        block_id, param_group = parse_param_group_name(name)

        if block_id is not None and param_group is not None:
            param_group_data.append({
                'block_id': block_id,
                'param_group': param_group,
                'avg_prune_ratio': info['prune_ratio'],
                'n_params': info['n_params'],
                'k': info['k'],
                'theta': info['theta'],
                'mean': info['mean'],
                'median': info['median'],
            })

    df = pd.DataFrame(param_group_data)

    if len(df) == 0:
        return df

    # 按参数组聚合（可能一个参数组有多个weight）
    agg_df = df.groupby(['block_id', 'param_group']).agg({
        'avg_prune_ratio': 'mean',
        'n_params': 'sum',
        'k': 'first',
        'theta': 'first',
        'mean': 'first',
        'median': 'first',
    }).reset_index()

    return agg_df


def plot_prune_ratio_heatmap(agg_df, output_dir, tau_star, global_prune_ratio):
    """绘制参数组剪枝比例热力图"""
    if len(agg_df) == 0:
        print("  警告: 没有参数组数据可用于绘制热力图")
        return

    output_dir = Path(output_dir)

    # 只保留 transformer blocks (block_id >= 0)
    blocks_df = agg_df[agg_df['block_id'] >= 0].copy()

    if len(blocks_df) == 0:
        print("  警告: 没有 transformer block 数据可用于绘制热力图")
        return

    # 创建透视表
    pivot_data = blocks_df.pivot(index='block_id', columns='param_group', values='avg_prune_ratio')

    # 按照逻辑顺序排列列
    column_order = ['attn_qkv', 'attn_proj', 'mlp_fc', 'mlp_proj', 'ln_1', 'ln_2']
    pivot_data = pivot_data[[col for col in column_order if col in pivot_data.columns]]

    # 转换为百分比
    pivot_data_pct = pivot_data * 100

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        pivot_data_pct,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',  # 反转颜色，红色表示高剪枝率
        cbar_kws={'label': 'Prune Ratio (%)'},
        vmin=0,
        vmax=max(pivot_data_pct.max().max() * 1.1, 15),  # 确保最小刻度为15%
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_xlabel('Parameter Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Block ID', fontsize=12, fontweight='bold')
    ax.set_title(f'Prune Ratio by Parameter Group (Global Target: {global_prune_ratio*100:.0f}%, τ*={tau_star:.2e})',
                 fontsize=14, fontweight='bold')

    # 更友好的列标签
    ax.set_xticklabels(['Attn QKV', 'Attn Proj', 'MLP FC', 'MLP Proj', 'LN 1', 'LN 2'])

    plt.tight_layout()
    heatmap_path = output_dir / 'param_group_prune_ratio_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 参数组剪枝比例热力图已保存: {heatmap_path}")
    plt.close()

    # 同时保存 CSV 数据
    csv_path = output_dir / 'param_group_prune_ratios.csv'
    pivot_data_pct.to_csv(csv_path)
    print(f"  ✓ 参数组剪枝比例 CSV 已保存: {csv_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='基于 Gamma 分布的自适应剪枝实验（纯一阶重要性）')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt',
                        help='检查点路径')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='累积的训练步数')
    parser.add_argument('--global_prune_ratio', type=float, default=0.1,
                        help='全局剪枝率 (0-1)')
    parser.add_argument('--eval_batches', type=int, default=10,
                        help='评估损失的批次数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='序列长度')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备')
    parser.add_argument('--output_dir', type=str,
                        default='results/gamma_adaptive_pruning_first_order',
                        help='输出目录')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("基于 Gamma 分布的自适应剪枝实验（纯一阶重要性）")
    print("=" * 80)
    print(f"全局剪枝率: {args.global_prune_ratio * 100:.1f}%")
    print(f"评估批次数: {args.eval_batches}")
    print(f"重要性公式: d_i = |g_i * θ_i| (纯一阶)")
    print("=" * 80)

    # 1. 加载模型
    print("\n[1/6] 加载模型...")
    model = get_gpt2_small(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ 模型已加载")

    # 2. 加载数据
    print("\n[2/6] 加载数据...")
    data_loader = get_wikitext103_dataloader(
        split='train',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0
    )

    # 3. 计算重要性得分（纯一阶）
    print("\n[3/6] 计算参数重要性（纯一阶）...")
    weights, gradients, exp_avg_sq = collect_gradients_and_momentum(
        model, data_loader, device, num_steps=args.num_steps
    )

    scores = compute_importance_scores_first_order(
        weights=weights,
        gradients=gradients,
        exp_avg_sq=exp_avg_sq,
        alpha=0.5  # 未使用，保持接口一致
    )
    print("✓ 重要性得分已计算（纯一阶）")

    # 4. 为每层拟合 Gamma 分布（排除 Embedding 层）
    print("\n[4/6] 为每层拟合 Gamma 分布（排除 Embedding 层）...")
    layer_info = []

    for name, score_tensor in tqdm(scores.items()):
        if 'weight' not in name:
            continue

        # 排除 Embedding 层
        if 'wte.weight' in name or 'wpe.weight' in name:
            print(f"  跳过 Embedding 层: {name}")
            continue

        scores_flat = score_tensor.flatten().numpy()
        k, theta = fit_gamma_distribution(scores_flat)

        layer_info.append({
            'name': name,
            'n_params': len(scores_flat),
            'k': k,
            'theta': theta,
            'scores': score_tensor,
            'mean': np.mean(scores_flat),
            'median': np.median(scores_flat),
        })

    print(f"✓ 已拟合 {len(layer_info)} 层（排除 Embedding）")

    # 5. 求解全局阈值并计算各层剪枝率
    print(f"\n[5/6] 求解全局阈值 (目标剪枝率: {args.global_prune_ratio*100:.1f}%)...")
    tau_star = solve_global_threshold(layer_info, args.global_prune_ratio)
    print(f"✓ 全局阈值 τ* = {tau_star:.6e}")

    layer_info = compute_layer_prune_ratios(layer_info, tau_star)

    # 打印各层剪枝率
    print("\n各层自适应剪枝率:")
    print(f"{'层名称':<50} {'参数数':<12} {'k':<10} {'θ':<12} {'剪枝率':<10}")
    print("-" * 100)

    for info in sorted(layer_info, key=lambda x: x['prune_ratio']):
        print(f"{info['name']:<50} {info['n_params']:<12,} {info['k']:<10.2f} "
              f"{info['theta']:<12.2e} {info['prune_ratio']*100:<10.2f}%")

    # 6. 评估剪枝前后的损失
    print(f"\n[6/6] 评估损失 ({args.eval_batches} 批次)...")

    # 评估原始模型
    print("  评估原始模型...")
    model_original = get_gpt2_small(pretrained=False)
    model_original.load_state_dict(checkpoint['model_state_dict'])
    loss_original = evaluate_loss(model_original, data_loader, device, args.eval_batches)
    print(f"  ✓ 原始损失: {loss_original:.4f}")

    # 应用剪枝
    print("  应用剪枝...")
    model_pruned = get_gpt2_small(pretrained=False)
    model_pruned.load_state_dict(checkpoint['model_state_dict'])
    model_pruned, mask_dict = apply_pruning(model_pruned, layer_info, tau_star)

    # 评估剪枝后模型
    print("  评估剪枝后模型...")
    loss_pruned = evaluate_loss(model_pruned, data_loader, device, args.eval_batches)
    print(f"  ✓ 剪枝后损失: {loss_pruned:.4f}")

    # 计算损失增加
    loss_increase = loss_pruned - loss_original
    loss_increase_pct = (loss_increase / loss_original) * 100

    print("\n" + "=" * 80)
    print("实验结果")
    print("=" * 80)
    print(f"全局剪枝率: {args.global_prune_ratio * 100:.1f}%")
    print(f"全局阈值 τ*: {tau_star:.6e}")
    print(f"重要性公式: d_i = |g_i * θ_i| (纯一阶)")
    print(f"\n原始损失:   {loss_original:.4f}")
    print(f"剪枝后损失: {loss_pruned:.4f}")
    print(f"损失增加:   {loss_increase:.4f} ({loss_increase_pct:+.2f}%)")

    # 统计各层剪枝率
    print("\n各层剪枝率统计:")
    prune_ratios = [info['prune_ratio'] for info in layer_info]
    print(f"  最小: {min(prune_ratios)*100:.2f}%")
    print(f"  最大: {max(prune_ratios)*100:.2f}%")
    print(f"  平均: {np.mean(prune_ratios)*100:.2f}%")
    print(f"  中位数: {np.median(prune_ratios)*100:.2f}%")

    # 保存结果
    df = pd.DataFrame(layer_info)
    df = df[['name', 'n_params', 'k', 'theta', 'mean', 'median', 'prune_ratio']]
    df.to_csv(output_dir / 'layer_prune_ratios.csv', index=False)
    print(f"\n✓ 层剪枝比例已保存到: {output_dir / 'layer_prune_ratios.csv'}")

    # 按参数组聚合并绘制热力图
    print("\n[生成可视化]")
    agg_df = aggregate_prune_ratios_by_param_group(layer_info)

    if len(agg_df) > 0:
        # 保存参数组聚合数据
        agg_df.to_csv(output_dir / 'param_group_aggregated.csv', index=False)
        print(f"  ✓ 参数组聚合数据已保存到: {output_dir / 'param_group_aggregated.csv'}")

        # 绘制热力图
        plot_prune_ratio_heatmap(agg_df, output_dir, tau_star, args.global_prune_ratio)

        # 打印参数组剪枝率统计
        print("\n参数组剪枝率统计:")
        for param_group in ['attn_qkv', 'attn_proj', 'mlp_fc', 'mlp_proj', 'ln_1', 'ln_2']:
            group_data = agg_df[agg_df['param_group'] == param_group]
            if len(group_data) > 0:
                ratios = group_data['avg_prune_ratio'].values * 100
                print(f"  {param_group:<12}: min={ratios.min():.2f}%, max={ratios.max():.2f}%, mean={ratios.mean():.2f}%")

    # 保存摘要
    summary = {
        'global_prune_ratio': args.global_prune_ratio,
        'tau_star': tau_star,
        'loss_original': loss_original,
        'loss_pruned': loss_pruned,
        'loss_increase': loss_increase,
        'loss_increase_pct': loss_increase_pct,
        'importance_formula': 'd_i = |g_i * θ_i| (first-order only)',
    }

    with open(output_dir / 'summary.txt', 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("=" * 80)


if __name__ == '__main__':
    main()
