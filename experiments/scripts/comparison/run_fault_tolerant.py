"""
容错训练模拟实验 (Fig 6)。

模拟多次从压缩 checkpoint 恢复的训练过程，展示累积误差。

运行命令:
    python experiments/scripts/comparison/run_fault_tolerant.py \
        --model gpt2-small \
        --total_steps 1000 \
        --num_recoveries 5 \
        --prune_ratio 0.5 \
        --methods "none,magnitude-uniform,first-order-uniform,second-order-gamma" \
        --num_importance_steps 50 \
        --batch_size 4 \
        --seq_length 512 \
        --lr 5e-5 \
        --alpha 0.5 \
        --device cuda \
        --output_dir results/fault_tolerant

实验流程:
    对每个方法独立运行完整训练流程：
    1. 加载预训练模型
    2. 训练 steps_per_segment 步
    3. 在恢复点压缩 checkpoint（计算重要性 + 剪枝）
    4. 继续训练下一段
    5. 重复直到完成所有恢复点
    6. 记录每个方法的 loss 曲线

方法对比:
    - none: 不压缩，作为 oracle baseline
    - magnitude-uniform: magnitude 得分 + uniform 分配
    - first-order-uniform: 一阶得分 + uniform 分配
    - second-order-gamma: 二阶得分 + gamma-adaptive 分配（我们的方法）
"""

import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

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
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small, get_gpt2_medium
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader
from src.ckpt_compress.methods.adam_prune.importance import (
    compute_importance_scores_abs,
    compute_importance_scores_first_order,
    compute_importance_scores_magnitude,
)


def should_exclude_param(name):
    """判断参数是否应该被排除（embedding 和 bias）。"""
    exclude_keywords = ['wte.weight', 'wpe.weight', 'bias']
    return any(keyword in name for keyword in exclude_keywords)


def collect_gradients(model, optimizer, data_loader, device, num_steps=50):
    """
    累积梯度和优化器状态。
    
    返回:
        weights: 权重字典
        gradients: 梯度字典
        exp_avg_sq: Adam 二阶矩字典
    """
    model.train()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    accumulated_gradients = defaultdict(lambda: 0)
    accumulated_exp_avg_sq = defaultdict(lambda: 0)

    print(f"  [收集梯度] 累积 {num_steps} 步...")
    data_iter = iter(data_loader)

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        
        # 处理不同模型的输出格式
        outputs = model(input_ids)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()

        # 累积梯度
        for name, param in model.named_parameters():
            if param.grad is not None and not should_exclude_param(name):
                accumulated_gradients[name] = accumulated_gradients[name] + param.grad.detach().cpu()

        optimizer.step()

        # 累积 exp_avg_sq
        for name, param in model.named_parameters():
            if param in optimizer.state and not should_exclude_param(name):
                exp_avg_sq = optimizer.state[param]['exp_avg_sq']
                accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] + exp_avg_sq.detach().cpu()

    # 平均
    for name in accumulated_gradients:
        accumulated_gradients[name] = accumulated_gradients[name] / num_steps
        accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] / num_steps

    weights = {name: param.detach().cpu() for name, param in model.named_parameters() if not should_exclude_param(name)}
    return weights, dict(accumulated_gradients), dict(accumulated_exp_avg_sq)


def fit_gamma_distribution(data):
    """拟合 Gamma 分布，返回 (k, θ)。"""
    data_positive = data[data > 0]
    if len(data_positive) == 0:
        return None, None

    try:
        k, loc, scale = stats.gamma.fit(data_positive, floc=0)
        return k, scale  # (shape, scale)
    except:
        return None, None


def solve_global_threshold(layer_info, global_prune_ratio):
    """
    求解全局阈值 τ*，使得全局剪枝率 = global_prune_ratio。
    
    方程: ρ = (1/N) Σ n_ℓ · P(k_ℓ, τ/θ_ℓ)
    """
    N = sum(info['n_params'] for info in layer_info)

    def objective(tau):
        total_pruned = 0
        for info in layer_info:
            k, theta = info['k'], info['theta']
            if k is None or theta is None:
                continue
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

    try:
        tau_star = bisect(objective, tau_min, tau_max, xtol=1e-10, maxiter=100)
    except ValueError:
        tau_max = np.percentile(all_scores, 90)
        tau_star = bisect(objective, tau_min, tau_max, xtol=1e-10, maxiter=100)

    return tau_star


def apply_uniform_pruning(model, optimizer, scores, prune_ratio):
    """
    Uniform 剪枝：对每层独立按得分排序，剪掉最低 p% 参数。
    """
    for name, param in model.named_parameters():
        if name not in scores or should_exclude_param(name):
            continue
        
        score = scores[name].to(param.device)
        flat_score = score.flatten()
        
        # 计算阈值
        k = int(len(flat_score) * prune_ratio)
        if k == 0:
            continue
        
        threshold = torch.kthvalue(flat_score, k).values
        
        # 创建掩码
        mask = (score > threshold).float()
        
        # 应用掩码到参数
        param.data.mul_(mask)
        
        # 应用掩码到优化器状态
        if param in optimizer.state:
            optimizer.state[param]['exp_avg'].mul_(mask)
            optimizer.state[param]['exp_avg_sq'].mul_(mask)


def apply_gamma_pruning(model, optimizer, scores, prune_ratio):
    """
    Gamma-adaptive 剪枝：拟合分布 + 求解全局阈值。
    """
    # 准备层信息
    layer_info = []
    for name, param in model.named_parameters():
        if name not in scores or should_exclude_param(name):
            continue
        
        score = scores[name].cpu().numpy()
        n_params = score.size
        
        # 拟合 Gamma 分布
        k, theta = fit_gamma_distribution(score.flatten())
        
        layer_info.append({
            'name': name,
            'n_params': n_params,
            'k': k,
            'theta': theta,
            'scores': score,
        })
    
    # 求解全局阈值
    tau_star = solve_global_threshold(layer_info, prune_ratio)
    
    # 应用剪枝
    for info in layer_info:
        name = info['name']
        scores_np = info['scores']
        
        if scores_np is None:
            continue
        
        # 创建掩码
        mask = (scores_np >= tau_star).astype(np.float32)
        mask_tensor = torch.from_numpy(mask)
        
        # 应用掩码到参数
        param = dict(model.named_parameters())[name]
        param.data.mul_(mask_tensor.to(param.device))
        
        # 应用掩码到优化器状态
        if param in optimizer.state:
            optimizer.state[param]['exp_avg'].mul_(mask_tensor.to(param.device))
            optimizer.state[param]['exp_avg_sq'].mul_(mask_tensor.to(param.device))


def train_one_step(model, optimizer, batch, device):
    """训练一步，返回 loss。"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    optimizer.zero_grad()
    
    # 处理不同模型的输出格式
    outputs = model(input_ids)
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model, val_batches, device):
    """评估模型，返回平均 loss。"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_batches:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 处理不同模型的输出格式
            outputs = model(input_ids)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(val_batches)


def run_method(method_name, args, train_loader, val_batches):
    """
    运行单个方法的完整训练流程。
    
    返回:
        loss_history: list of (global_step, loss, perplexity)
    """
    print(f"\n{'='*60}")
    print(f"运行方法: {method_name}")
    print(f"{'='*60}")
    
    # 加载模型
    if args.model == 'gpt2-small':
        model = get_gpt2_small(pretrained=True)
    else:  # gpt2-medium
        model = get_gpt2_medium(pretrained=True)
    
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    steps_per_segment = args.total_steps // (args.num_recoveries + 1)
    loss_history = []
    global_step = 0
    
    data_iter = iter(train_loader)
    
    for recovery in range(args.num_recoveries + 1):
        print(f"\n--- Segment {recovery + 1}/{args.num_recoveries + 1} ---")
        
        # 训练 steps_per_segment 步
        for step in tqdm(range(steps_per_segment), desc=f"Training segment {recovery + 1}"):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            loss = train_one_step(model, optimizer, batch, args.device)
            global_step += 1
            
            # 每 10 步评估一次
            if step % 10 == 0:
                val_loss = evaluate(model, val_batches, args.device)
                perplexity = np.exp(val_loss)
                loss_history.append((global_step, val_loss, perplexity))
                
                if step % 50 == 0:
                    print(f"  Step {global_step}: Loss={val_loss:.4f}, PPL={perplexity:.2f}")
        
        # 在恢复点压缩 checkpoint（最后一段不压缩）
        if recovery < args.num_recoveries and method_name != "none":
            print(f"\n  [恢复点 {recovery + 1}] 压缩 checkpoint...")
            
            # 计算重要性得分
            weights, gradients, exp_avg_sq = collect_gradients(
                model, optimizer, train_loader, args.device, args.num_importance_steps
            )
            
            # 根据方法计算得分
            if "magnitude" in method_name:
                scores = compute_importance_scores_magnitude(weights)
            elif "first-order" in method_name:
                scores = compute_importance_scores_first_order(weights, gradients, exp_avg_sq)
            else:  # second-order
                scores = compute_importance_scores_abs(weights, gradients, exp_avg_sq, alpha=args.alpha)
            
            # 应用剪枝
            if "uniform" in method_name:
                apply_uniform_pruning(model, optimizer, scores, args.prune_ratio)
            else:  # gamma
                apply_gamma_pruning(model, optimizer, scores, args.prune_ratio)
            
            # 评估剪枝后的损失
            val_loss = evaluate(model, val_batches, args.device)
            perplexity = np.exp(val_loss)
            print(f"  剪枝后: Loss={val_loss:.4f}, PPL={perplexity:.2f}")
    
    return loss_history


def plot_results(results, args):
    """绘制结果图。"""
    plt.figure(figsize=(12, 6))
    
    colors = {
        'none': 'black',
        'magnitude-uniform': 'blue',
        'first-order-uniform': 'green',
        'second-order-gamma': 'red',
    }
    
    for method_name, loss_history in results.items():
        steps = [x[0] for x in loss_history]
        losses = [x[1] for x in loss_history]
        
        plt.plot(steps, losses, label=method_name, color=colors.get(method_name, 'gray'), linewidth=2)
    
    # 画恢复点的竖直虚线
    steps_per_segment = args.total_steps // (args.num_recoveries + 1)
    for i in range(1, args.num_recoveries + 1):
        recovery_step = i * steps_per_segment
        plt.axvline(x=recovery_step, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title(f'Fault-Tolerant Training: {args.model.upper()} on WikiText-103\n'
              f'({args.num_recoveries} recoveries, prune_ratio={args.prune_ratio})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'fault_tolerant_loss_curve.png'
    plt.savefig(plot_path, dpi=300)
    print(f"\n图片已保存: {plot_path}")
    plt.close()


def save_csv(results, args):
    """保存 CSV 结果。"""
    rows = []
    for method_name, loss_history in results.items():
        for step, loss, ppl in loss_history:
            rows.append({
                'method': method_name,
                'global_step': step,
                'loss': loss,
                'perplexity': ppl,
            })
    
    df = pd.DataFrame(rows)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'fault_tolerant_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV 已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='容错训练模拟实验')
    parser.add_argument('--model', type=str, default='gpt2-small', choices=['gpt2-small', 'gpt2-medium'],
                        help='模型类型')
    parser.add_argument('--total_steps', type=int, default=1000,
                        help='总训练步数')
    parser.add_argument('--num_recoveries', type=int, default=5,
                        help='恢复次数')
    parser.add_argument('--prune_ratio', type=float, default=0.5,
                        help='剪枝比例')
    parser.add_argument('--methods', type=str, default='none,magnitude-uniform,first-order-uniform,second-order-gamma',
                        help='方法列表（逗号分隔）')
    parser.add_argument('--num_importance_steps', type=int, default=50,
                        help='计算重要性的步数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='序列长度')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='学习率')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='二阶项权重系数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--output_dir', type=str, default='results/fault_tolerant',
                        help='输出目录')
    
    args = parser.parse_args()
    
    print("="*60)
    print("容错训练模拟实验")
    print("="*60)
    print(f"模型: {args.model}")
    print(f"总步数: {args.total_steps}")
    print(f"恢复次数: {args.num_recoveries}")
    print(f"剪枝比例: {args.prune_ratio}")
    print(f"方法: {args.methods}")
    print("="*60)
    
    # 加载数据
    print("\n加载数据...")
    train_loader = get_wikitext103_dataloader(
        split='train',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0,
        shuffle=True,
    )
    
    val_loader = get_wikitext103_dataloader(
        split='validation',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0,
        shuffle=False,
    )
    
    # 缓存验证批次
    print("缓存验证批次...")
    val_batches = []
    for i, batch in enumerate(val_loader):
        val_batches.append(batch)
        if i >= 9:  # 10 个批次
            break
    
    # 运行每个方法
    methods = [m.strip() for m in args.methods.split(',')]
    results = {}
    
    for method_name in methods:
        loss_history = run_method(method_name, args, train_loader, val_batches)
        results[method_name] = loss_history
    
    # 保存结果
    print("\n保存结果...")
    save_csv(results, args)
    plot_results(results, args)
    
    print("\n实验完成！")


if __name__ == '__main__':
    main()
