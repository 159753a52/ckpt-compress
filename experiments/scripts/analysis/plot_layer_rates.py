#!/usr/bin/env python3
"""
Fig 5: 逐层剪枝率分布

柱状图展示 Gamma-adaptive 分配的逐层剪枝率 vs Uniform 的水平线。

用法:
    python plot_layer_rates.py --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import bisect
from tqdm import tqdm
from typing import Dict, Tuple

from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader
from src.ckpt_compress.methods.adam_prune.importance import compute_importance_scores_abs


def parse_args():
    parser = argparse.ArgumentParser(description='Fig 5: 逐层剪枝率分布')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt',
        help='检查点路径'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=100,
        help='用于计算梯度的步数'
    )
    parser.add_argument(
        '--global_prune_ratio',
        type=float,
        default=0.3,
        help='全局剪枝率'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='批次大小'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='二阶项权重系数'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备 (cuda/cpu)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/figures',
        help='输出目录'
    )
    return parser.parse_args()


def accumulate_gradients(model, dataloader, num_steps, device):
    """
    累积梯度和 exp_avg_sq。
    
    返回:
        gradients: Dict[str, Tensor]
        exp_avg_sq: Dict[str, Tensor]
    """
    model.train()
    
    # 初始化累积器
    gradients = {}
    exp_avg_sq = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradients[name] = torch.zeros_like(param)
            exp_avg_sq[name] = torch.zeros_like(param)
    
    # 累积梯度
    step_count = 0
    for batch in tqdm(dataloader, desc='Accumulating gradients', total=num_steps):
        if step_count >= num_steps:
            break
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        logits = model(input_ids)
        
        # 计算损失
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # 反向传播
        model.zero_grad()
        loss.backward()
        
        # 累积梯度和平方梯度
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients[name] += param.grad.detach()
                exp_avg_sq[name] += param.grad.detach() ** 2
        
        step_count += 1
    
    # 平均
    for name in gradients:
        gradients[name] /= step_count
        exp_avg_sq[name] /= step_count
    
    return gradients, exp_avg_sq


def fit_gamma_distributions(scores_dict) -> Dict[str, Tuple[float, float, int]]:
    """
    对每层拟合 Gamma 分布。
    
    返回:
        Dict[layer_name, (k, theta, n_params)]
    """
    gamma_params = {}
    
    for name, scores in scores_dict.items():
        scores_flat = scores.cpu().numpy().flatten()
        scores_positive = scores_flat[scores_flat > 0]
        
        if len(scores_positive) == 0:
            continue
        
        try:
            # 拟合 Gamma 分布
            k, _, theta = stats.gamma.fit(scores_positive, floc=0)
            n_params = len(scores_flat)
            gamma_params[name] = (k, theta, n_params)
        except Exception as e:
            print(f"Warning: Failed to fit Gamma for {name}: {e}")
            continue
    
    return gamma_params


def find_global_threshold(gamma_params, global_prune_ratio):
    """
    使用二分法找到全局阈值 c*，使得总剪枝率 = global_prune_ratio。
    
    目标: sum(n_l * gamma.cdf(c, k_l, scale=theta_l)) = P * sum(n_l)
    """
    # 计算总参数数
    total_params = sum(n for _, _, n in gamma_params.values())
    target_pruned = global_prune_ratio * total_params
    
    # 定义目标函数
    def objective(c):
        pruned = 0
        for k, theta, n in gamma_params.values():
            pruned += n * stats.gamma.cdf(c, k, scale=theta)
        return pruned - target_pruned
    
    # 找到合适的搜索范围
    # 收集所有层的 scores 范围
    all_scores = []
    for k, theta, _ in gamma_params.values():
        # 使用 Gamma 分布的 ppf 来估计范围
        all_scores.append(stats.gamma.ppf(0.01, k, scale=theta))
        all_scores.append(stats.gamma.ppf(0.99, k, scale=theta))
    
    c_min = min(all_scores)
    c_max = max(all_scores)
    
    # 二分法求解
    try:
        c_star = bisect(objective, c_min, c_max, xtol=1e-6)
        return c_star
    except ValueError as e:
        print(f"Warning: Bisection failed: {e}")
        # 如果二分法失败，使用简单的线性搜索
        c_values = np.linspace(c_min, c_max, 1000)
        errors = [abs(objective(c)) for c in c_values]
        return c_values[np.argmin(errors)]


def compute_layer_prune_rates(gamma_params, c_star):
    """
    计算每层的自适应剪枝率。
    
    p_l = gamma.cdf(c*, k_l, scale=theta_l)
    """
    layer_rates = {}
    
    for name, (k, theta, n) in gamma_params.items():
        prune_rate = stats.gamma.cdf(c_star, k, scale=theta)
        layer_rates[name] = prune_rate
    
    return layer_rates


def simplify_layer_name(name):
    """
    简化层名称用于显示。
    
    transformer.h.0.attn.c_attn.weight → B0.QKV
    transformer.h.0.attn.c_proj.weight → B0.AProj
    transformer.h.0.mlp.c_fc.weight → B0.FC
    transformer.h.0.mlp.c_proj.weight → B0.MProj
    transformer.h.0.ln_1.weight → B0.LN1
    transformer.h.0.ln_2.weight → B0.LN2
    """
    # 移除 'transformer.h.' 前缀
    if name.startswith('transformer.h.'):
        name = name[len('transformer.h.'):]
    
    # 提取 block 编号
    parts = name.split('.')
    if len(parts) < 2:
        return name
    
    block_num = parts[0]
    rest = '.'.join(parts[1:])
    
    # 映射层类型
    if 'attn.c_attn' in rest:
        layer_type = 'QKV'
    elif 'attn.c_proj' in rest:
        layer_type = 'AProj'
    elif 'mlp.c_fc' in rest:
        layer_type = 'FC'
    elif 'mlp.c_proj' in rest:
        layer_type = 'MProj'
    elif 'ln_1' in rest:
        layer_type = 'LN1'
    elif 'ln_2' in rest:
        layer_type = 'LN2'
    else:
        layer_type = rest.replace('.weight', '').replace('.bias', '')
    
    return f"B{block_num}.{layer_type}"


def get_layer_color(name):
    """
    根据层类型返回颜色。
    
    Attention: 蓝色
    MLP: 橙色
    LayerNorm: 绿色
    """
    if 'attn' in name or 'QKV' in name or 'AProj' in name:
        return 'steelblue'
    elif 'mlp' in name or 'FC' in name or 'MProj' in name:
        return 'darkorange'
    elif 'ln' in name or 'LN' in name:
        return 'forestgreen'
    else:
        return 'gray'


def plot_layer_rates(layer_rates, global_prune_ratio, output_dir):
    """
    画逐层剪枝率柱状图。
    """
    # 准备数据
    layer_names = []
    prune_rates = []
    colors = []
    
    for name, rate in sorted(layer_rates.items()):
        layer_names.append(simplify_layer_name(name))
        prune_rates.append(rate * 100)  # 转为百分比
        colors.append(get_layer_color(name))
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # 画柱状图
    x = np.arange(len(layer_names))
    bars = ax.bar(x, prune_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 画 Uniform 水平线
    ax.axhline(y=global_prune_ratio * 100, color='red', linestyle='--', 
               linewidth=2, label=f'Uniform ({global_prune_ratio*100:.1f}%)')
    
    # 设置标签
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pruning Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Layer-wise Pruning Rates: Gamma-Adaptive vs Uniform', 
                 fontsize=14, fontweight='bold')
    
    # 设置 X 轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', edgecolor='black', label='Attention'),
        Patch(facecolor='darkorange', edgecolor='black', label='MLP'),
        Patch(facecolor='forestgreen', edgecolor='black', label='LayerNorm'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Uniform')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 添加网格
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / 'layer_pruning_rates.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'layer_pruning_rates.pdf', bbox_inches='tight')
    
    print(f"Saved figures to {output_dir}")
    plt.close(fig)


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print("Loading model...")
    model = get_gpt2_small(pretrained=False)
    model = model.to(device)
    
    # 加载检查点
    if Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found, using random weights")
    
    # 加载数据
    print("Loading data...")
    dataloader = get_wikitext103_dataloader(
        split='train',
        batch_size=args.batch_size,
        seq_length=512,
        num_workers=2
    )
    
    # 累积梯度
    print(f"Accumulating gradients for {args.num_steps} steps...")
    gradients, exp_avg_sq = accumulate_gradients(model, dataloader, args.num_steps, device)
    
    # 获取权重
    weights = {name: param.detach() for name, param in model.named_parameters()}
    
    # 计算重要性得分
    print("Computing importance scores...")
    scores = compute_importance_scores_abs(weights, gradients, exp_avg_sq, alpha=args.alpha)
    
    # 拟合 Gamma 分布
    print("Fitting Gamma distributions...")
    gamma_params = fit_gamma_distributions(scores)
    print(f"Successfully fitted {len(gamma_params)} layers")
    
    # 找到全局阈值
    print(f"Finding global threshold for {args.global_prune_ratio*100}% pruning...")
    c_star = find_global_threshold(gamma_params, args.global_prune_ratio)
    print(f"Global threshold c* = {c_star:.6f}")
    
    # 计算每层剪枝率
    print("Computing layer-wise pruning rates...")
    layer_rates = compute_layer_prune_rates(gamma_params, c_star)
    
    # 验证总剪枝率
    total_params = sum(n for _, _, n in gamma_params.values())
    total_pruned = sum(layer_rates[name] * gamma_params[name][2] for name in layer_rates)
    actual_prune_ratio = total_pruned / total_params
    print(f"Actual global pruning rate: {actual_prune_ratio*100:.2f}%")
    
    # 画图
    print("Plotting...")
    plot_layer_rates(layer_rates, args.global_prune_ratio, args.output_dir)
    
    print("Done!")


if __name__ == '__main__':
    main()
