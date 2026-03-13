#!/usr/bin/env python3
"""
Fig 3: Gamma 拟合质量验证

选择 3 个代表性层（Attention QKV, MLP FC, LayerNorm），
画 damage score 直方图 + 拟合的 Gamma PDF 曲线。

用法:
    python plot_gamma_fitting.py --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt
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
from tqdm import tqdm

from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader
from src.ckpt_compress.methods.adam_prune.importance import compute_importance_scores_abs


def parse_args():
    parser = argparse.ArgumentParser(description='Fig 3: Gamma 拟合质量验证')
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


def fit_gamma_and_test(scores_flat):
    """
    拟合 Gamma 分布并进行 KS 检验。
    
    返回:
        k: shape parameter
        theta: scale parameter
        ks_stat: KS 统计量
        p_value: p-value
    """
    # 只使用正值
    scores_positive = scores_flat[scores_flat > 0]
    
    if len(scores_positive) == 0:
        return None, None, None, None
    
    # 拟合 Gamma 分布
    k, _, theta = stats.gamma.fit(scores_positive, floc=0)
    
    # KS 检验
    ks_stat, p_value = stats.kstest(scores_positive, 'gamma', args=(k, 0, theta))
    
    return k, theta, ks_stat, p_value


def plot_gamma_fitting(scores_dict, output_dir):
    """
    画 3 个代表性层的 Gamma 拟合图。
    """
    # 选择 3 个代表性层
    layer_configs = [
        {
            'key': 'transformer.h.0.attn.c_attn.weight',
            'title': 'Attn QKV (Block 0)',
            'color': 'steelblue'
        },
        {
            'key': 'transformer.h.0.mlp.c_fc.weight',
            'title': 'MLP FC (Block 0)',
            'color': 'steelblue'
        },
        {
            'key': 'transformer.h.0.ln_1.weight',
            'title': 'LayerNorm (Block 0)',
            'color': 'steelblue'
        }
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, config in enumerate(layer_configs):
        ax = axes[idx]
        key = config['key']
        
        if key not in scores_dict:
            print(f"Warning: {key} not found in scores_dict")
            continue
        
        # 获取 scores 并转为 numpy
        scores_flat = scores_dict[key].cpu().numpy().flatten()
        scores_positive = scores_flat[scores_flat > 0]
        
        if len(scores_positive) == 0:
            print(f"Warning: No positive scores for {key}")
            continue
        
        # 拟合 Gamma 分布
        k, theta, ks_stat, p_value = fit_gamma_and_test(scores_flat)
        
        if k is None:
            print(f"Warning: Failed to fit Gamma for {key}")
            continue
        
        # 画直方图
        ax.hist(scores_positive, bins=100, density=True, alpha=0.7, 
                color=config['color'], label='Empirical')
        
        # 画 Gamma PDF 曲线
        x = np.linspace(scores_positive.min(), scores_positive.max(), 1000)
        pdf = stats.gamma.pdf(x, k, scale=theta)
        ax.plot(x, pdf, 'r-', linewidth=2, label='Gamma PDF')
        
        # 设置标题和标签
        ax.set_title(config['title'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Damage Score', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        
        # 添加统计信息
        textstr = f'k={k:.3f}, θ={theta:.2e}\nKS p={p_value:.3f}'
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / 'gamma_fitting_quality.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'gamma_fitting_quality.pdf', bbox_inches='tight')
    
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
    
    # 画图
    print("Plotting...")
    plot_gamma_fitting(scores, args.output_dir)
    
    print("Done!")


if __name__ == '__main__':
    main()
