"""
消融实验脚本 - 拆解两个贡献的独立增益（Table 2）

5 种组合：
A: Magnitude + Uniform
B: First-order + Uniform  
C: Second-order (1st+2nd) + Uniform
D: First-order + Gamma-adaptive
E: Second-order (1st+2nd) + Gamma-adaptive (Full)

运行命令:
    python experiments/scripts/comparison/run_ablation.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --model gpt2-small \
        --prune_ratios 0.1,0.2,0.3,0.5,0.7 \
        --num_steps 100 \
        --eval_batches 10 \
        --batch_size 4 \
        --alpha 0.5 \
        --device cuda \
        --output_dir results/ablation
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
from datetime import datetime
import argparse
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small, get_gpt2_medium
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader
from src.ckpt_compress.methods.adam_prune.importance import (
    compute_importance_scores_abs,
    compute_importance_scores_first_order,
    compute_importance_scores_magnitude,
)


def cache_batches(data_loader, num_batches):
    """缓存数据批次到内存"""
    cached = []
    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break
        cached.append({
            'input_ids': batch['input_ids'].clone(),
            'labels': batch['labels'].clone()
        })
    return cached


def collect_gradients_and_momentum(model, cached_train_batches, device, num_steps=100):
    """累积梯度和动量（使用缓存批次）"""
    model.train()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    accumulated_gradients = defaultdict(lambda: 0)
    accumulated_exp_avg_sq = defaultdict(lambda: 0)

    print(f"累积 {num_steps} 步的梯度和动量...")

    num_batches = len(cached_train_batches)

    for step in tqdm(range(num_steps)):
        batch = cached_train_batches[step % num_batches]

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


def filter_prunable_params(scores_dict):
    """排除 embedding 和 bias 参数"""
    filtered = {}
    for name, scores in scores_dict.items():
        # 排除 embedding (wte.weight, wpe.weight) 和 bias
        if 'wte.weight' in name or 'wpe.weight' in name or 'bias' in name:
            continue
        # 只保留 weight 参数
        if 'weight' in name:
            filtered[name] = scores
    return filtered


def uniform_pruning(scores_dict, prune_ratio):
    """Uniform 剪枝：每层独立按得分排序，剪掉最低 p%"""
    masks = {}
    total_params = 0
    total_pruned = 0
    
    for name, scores in scores_dict.items():
        n_params = scores.numel()
        n_prune = int(n_params * prune_ratio)
        
        # 找到阈值
        flat_scores = scores.flatten()
        threshold = torch.kthvalue(flat_scores, n_prune + 1)[0] if n_prune < n_params else float('inf')
        
        # 创建 mask（1=保留，0=剪枝）
        mask = (scores > threshold).float()
        masks[name] = mask
        
        total_params += n_params
        total_pruned += (mask == 0).sum().item()
    
    actual_ratio = total_pruned / total_params if total_params > 0 else 0.0
    return masks, actual_ratio


def fit_gamma_distribution(data):
    """拟合 Gamma 分布，返回 (k, θ)"""
    data_positive = data[data > 0]
    if len(data_positive) == 0:
        return None, None

    try:
        k, loc, scale = stats.gamma.fit(data_positive, floc=0)
        return k, scale
    except:
        return None, None


def gamma_adaptive_pruning(scores_dict, global_prune_ratio):
    """Gamma-adaptive 剪枝：拟合 Gamma，二分法求 τ*"""
    # 1. 为每层拟合 Gamma 分布
    layer_info = []
    for name, scores in scores_dict.items():
        flat_scores = scores.flatten().numpy()
        k, theta = fit_gamma_distribution(flat_scores)
        layer_info.append({
            'name': name,
            'scores': flat_scores,
            'n_params': len(flat_scores),
            'k': k,
            'theta': theta
        })
    
    # 2. 二分法求解全局阈值 τ*
    N = sum(info['n_params'] for info in layer_info)
    
    def objective(tau):
        total_pruned = 0
        for info in layer_info:
            k, theta = info['k'], info['theta']
            if k is None or theta is None:
                continue
            cdf_value = stats.gamma.cdf(tau, k, scale=theta)
            total_pruned += info['n_params'] * cdf_value
        return total_pruned / N - global_prune_ratio
    
    # 确定搜索范围
    all_scores = []
    for info in layer_info:
        if info['scores'] is not None:
            all_scores.extend(info['scores'].tolist())
    
    all_scores = np.array(all_scores)
    tau_min = np.percentile(all_scores, 0.1)
    tau_max = np.percentile(all_scores, 50)
    
    try:
        tau_star = bisect(objective, tau_min, tau_max, xtol=1e-10, maxiter=100)
    except ValueError:
        tau_max = np.percentile(all_scores, 90)
        try:
            tau_star = bisect(objective, tau_min, tau_max, xtol=1e-10, maxiter=100)
        except ValueError:
            tau_max = np.percentile(all_scores, 99)
            tau_star = bisect(objective, tau_min, tau_max, xtol=1e-10, maxiter=100)
    
    # 3. 根据 τ* 计算各层剪枝率并生成 mask
    masks = {}
    total_params = 0
    total_pruned = 0
    
    for info in layer_info:
        name = info['name']
        scores = scores_dict[name]
        k, theta = info['k'], info['theta']
        
        if k is None or theta is None:
            # 无法拟合，不剪枝
            masks[name] = torch.ones_like(scores)
        else:
            # 使用 τ* 作为阈值
            mask = (scores > tau_star).float()
            masks[name] = mask
            
            total_params += info['n_params']
            total_pruned += (mask == 0).sum().item()
    
    actual_ratio = total_pruned / total_params if total_params > 0 else 0.0
    return masks, actual_ratio


def apply_masks_and_evaluate(model, masks, cached_eval_batches, device):
    """应用 mask 并评估损失"""
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 保存原始权重
    original_weights = {}
    for name, param in model.named_parameters():
        if name in masks:
            original_weights[name] = param.data.clone()
    
    # 应用 mask（原地修改）
    for name, param in model.named_parameters():
        if name in masks:
            param.data.mul_(masks[name].to(device))
    
    # 评估
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in cached_eval_batches:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # 恢复原始权重
    for name, param in model.named_parameters():
        if name in original_weights:
            param.data.copy_(original_weights[name])
    
    return avg_loss


def evaluate_original_loss(model, cached_eval_batches, device):
    """评估原始模型损失"""
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in cached_eval_batches:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def run_ablation_experiment(args):
    """运行消融实验"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载模型
    print("\n[1/7] 加载模型...")
    if args.model == 'gpt2-small':
        model = get_gpt2_small(pretrained=False)
    elif args.model == 'gpt2-medium':
        model = get_gpt2_medium(pretrained=False)
    else:
        raise ValueError(f"不支持的模型: {args.model}")
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ 模型已加载")
    
    # 2. 加载数据
    print("\n[2/7] 加载数据...")
    train_loader = get_wikitext103_dataloader(
        split='train',
        batch_size=args.batch_size,
        seq_length=512,
        num_workers=0
    )
    
    eval_loader = get_wikitext103_dataloader(
        split='validation',
        batch_size=args.batch_size,
        seq_length=512,
        num_workers=0
    )
    
    # 缓存批次
    num_train_batches = max(args.num_steps, 200)
    cached_train_batches = cache_batches(train_loader, num_train_batches)
    cached_eval_batches = cache_batches(eval_loader, args.eval_batches)
    print(f"✓ 已缓存 {len(cached_train_batches)} 个训练批次，{len(cached_eval_batches)} 个评估批次")
    
    # 3. 累积梯度和动量（一次，共用）
    print("\n[3/7] 累积梯度和动量...")
    weights, gradients, exp_avg_sq = collect_gradients_and_momentum(
        model, cached_train_batches, device, args.num_steps
    )
    print("✓ 梯度和动量已累积")
    
    # 4. 计算 3 种重要性得分
    print("\n[4/7] 计算重要性得分...")
    magnitude_scores = compute_importance_scores_magnitude(weights)
    first_order_scores = compute_importance_scores_first_order(weights, gradients, exp_avg_sq, args.alpha)
    second_order_scores = compute_importance_scores_abs(weights, gradients, exp_avg_sq, args.alpha)
    
    # 过滤可剪枝参数
    magnitude_scores = filter_prunable_params(magnitude_scores)
    first_order_scores = filter_prunable_params(first_order_scores)
    second_order_scores = filter_prunable_params(second_order_scores)
    print("✓ 重要性得分已计算")
    
    # 5. 评估原始损失
    print("\n[5/7] 评估原始损失...")
    loss_original = evaluate_original_loss(model, cached_eval_batches, device)
    print(f"✓ 原始损失: {loss_original:.4f}")
    
    # 6. 运行 5 种组合
    print("\n[6/7] 运行消融实验...")
    prune_ratios = [float(r) for r in args.prune_ratios.split(',')]
    
    results = []
    
    combinations = [
        ('A', 'Magnitude', 'Uniform', magnitude_scores, uniform_pruning),
        ('B', 'First-order', 'Uniform', first_order_scores, uniform_pruning),
        ('C', 'Second-order', 'Uniform', second_order_scores, uniform_pruning),
        ('D', 'First-order', 'Gamma-adaptive', first_order_scores, gamma_adaptive_pruning),
        ('E', 'Second-order', 'Gamma-adaptive', second_order_scores, gamma_adaptive_pruning),
    ]
    
    for combo_id, importance_name, allocation_name, scores, pruning_fn in combinations:
        print(f"\n组合 {combo_id}: {importance_name} + {allocation_name}")
        
        for prune_ratio in prune_ratios:
            print(f"  剪枝率 {prune_ratio:.1%}...", end=' ')
            
            # 剪枝
            if pruning_fn == uniform_pruning:
                masks, actual_ratio = pruning_fn(scores, prune_ratio)
            else:  # gamma_adaptive_pruning
                masks, actual_ratio = pruning_fn(scores, prune_ratio)
            
            # 评估
            loss_pruned = apply_masks_and_evaluate(model, masks, cached_eval_batches, device)
            loss_increase = loss_pruned - loss_original
            loss_increase_pct = (loss_increase / loss_original) * 100 if loss_original > 0 else 0.0
            
            print(f"实际剪枝率={actual_ratio:.1%}, 损失增加={loss_increase:.4f} ({loss_increase_pct:.2f}%)")
            
            results.append({
                'combination': combo_id,
                'importance': importance_name,
                'allocation': allocation_name,
                'prune_ratio': prune_ratio,
                'actual_ratio': actual_ratio,
                'loss_original': loss_original,
                'loss_pruned': loss_pruned,
                'loss_increase': loss_increase,
                'loss_increase_pct': loss_increase_pct
            })
    
    # 7. 保存结果
    print("\n[7/7] 保存结果...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f'ablation_{args.model}_{timestamp}.csv'
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"✓ 结果已保存到: {csv_path}")
    
    # 打印汇总表
    print("\n" + "="*80)
    print("消融实验结果汇总")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='消融实验 - 拆解两个贡献的独立增益')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--model', type=str, default='gpt2-small', choices=['gpt2-small', 'gpt2-medium'], help='模型类型')
    parser.add_argument('--prune_ratios', type=str, default='0.1,0.2,0.3,0.5,0.7', help='剪枝率列表（逗号分隔）')
    parser.add_argument('--num_steps', type=int, default=100, help='累积梯度的步数')
    parser.add_argument('--eval_batches', type=int, default=10, help='评估批次数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--alpha', type=float, default=0.5, help='二阶项权重系数')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--output_dir', type=str, default='results/ablation', help='输出目录')
    
    args = parser.parse_args()
    run_ablation_experiment(args)


if __name__ == '__main__':
    main()
