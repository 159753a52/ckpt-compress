"""
统一实验框架：论文 Table 1 核心实验。

在固定剪枝比例下对比多种方法的质量：
- 重要性方法: magnitude, first-order, second-order
- 分配策略: uniform, gamma-adaptive

运行命令:
    python experiments/scripts/comparison/run_main_experiment.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --model gpt2-small \
        --importance second-order \
        --allocation gamma-adaptive \
        --prune_ratios 0.1,0.2,0.3,0.5,0.7,0.9 \
        --num_steps 100 \
        --eval_batches 10 \
        --batch_size 4 \
        --seq_length 512 \
        --alpha 0.5 \
        --device cuda \
        --output_dir results/paper_results/table1

实验流程:
1. 加载模型和检查点
2. 缓存训练批次和验证批次
3. 累积梯度和 Adam exp_avg_sq
4. 根据 --importance 计算重要性得分
5. 对每个 prune_ratio:
   - uniform: 对所有层施加相同剪枝率
   - gamma-adaptive: 拟合 Gamma 分布，求全局阈值
6. 评估剪枝后的 loss
7. 输出 CSV 结果和 JSON 摘要
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
import json
import argparse
import copy

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small, get_gpt2_medium
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader
from src.ckpt_compress.methods.adam_prune.importance import (
    compute_importance_scores_abs,
    compute_importance_scores_first_order,
    compute_importance_scores_magnitude,
)


def cache_batches(data_loader, num_batches):
    """
    缓存数据批次到内存。
    
    参数:
        data_loader: PyTorch DataLoader
        num_batches: 要缓存的批次数量
    
    返回:
        缓存的批次列表
    """
    cached = []
    data_iter = iter(data_loader)
    for _ in range(num_batches):
        try:
            batch = next(data_iter)
            cached.append({
                'input_ids': batch['input_ids'].clone(),
                'labels': batch['labels'].clone()
            })
        except StopIteration:
            break
    return cached


def collect_gradients_and_momentum(model, cached_train_batches, device, num_steps=100, model_type='gpt2-small'):
    """
    使用缓存的训练批次累积梯度和动量。
    
    参数:
        model: PyTorch 模型
        cached_train_batches: 缓存的训练批次
        device: 设备 (cuda/cpu)
        num_steps: 累积步数
        model_type: 模型类型 (gpt2-small/gpt2-medium)
    
    返回:
        (weights, gradients, exp_avg_sq) 三元组
    """
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
        
        # 根据模型类型处理输出
        if model_type == 'gpt2-small':
            logits = model(input_ids)
        else:  # gpt2-medium
            outputs = model(input_ids)
            logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()

        # 累积梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                accumulated_gradients[name] = accumulated_gradients[name] + param.grad.detach().cpu()

        optimizer.step()

        # 累积 exp_avg_sq
        for name, param in model.named_parameters():
            if name in optimizer.state[param]:
                exp_avg_sq = optimizer.state[param]['exp_avg_sq']
                accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] + exp_avg_sq.detach().cpu()

    # 计算平均值
    for name in accumulated_gradients:
        accumulated_gradients[name] = accumulated_gradients[name] / num_steps
        accumulated_exp_avg_sq[name] = accumulated_exp_avg_sq[name] / num_steps

    weights = {name: param.detach().cpu() for name, param in model.named_parameters()}
    return weights, dict(accumulated_gradients), dict(accumulated_exp_avg_sq)


def evaluate_loss_with_cached_batches(model, cached_batches, device, model_type='gpt2-small'):
    """
    使用缓存批次评估损失。
    
    参数:
        model: PyTorch 模型
        cached_batches: 缓存的批次
        device: 设备
        model_type: 模型类型
    
    返回:
        平均损失值
    """
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    with torch.no_grad():
        for batch in cached_batches:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 根据模型类型处理输出
            if model_type == 'gpt2-small':
                logits = model(input_ids)
            else:  # gpt2-medium
                outputs = model(input_ids)
                logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()

    return total_loss / len(cached_batches)


def filter_prunable_params(weights, gradients, exp_avg_sq):
    """
    过滤出可剪枝的参数（排除 embedding 和 bias）。
    
    参数:
        weights: 权重字典
        gradients: 梯度字典
        exp_avg_sq: Adam 二阶矩字典
    
    返回:
        (filtered_weights, filtered_gradients, filtered_exp_avg_sq) 三元组
    """
    filtered_weights = {}
    filtered_gradients = {}
    filtered_exp_avg_sq = {}
    
    for name in weights:
        # 排除 embedding 层和 bias 参数
        if 'wte.weight' in name or 'wpe.weight' in name or 'bias' in name:
            continue
        
        # 只保留 weight 参数
        if 'weight' not in name:
            continue
        
        filtered_weights[name] = weights[name]
        filtered_gradients[name] = gradients.get(name, torch.zeros_like(weights[name]))
        filtered_exp_avg_sq[name] = exp_avg_sq.get(name, torch.zeros_like(weights[name]))
    
    return filtered_weights, filtered_gradients, filtered_exp_avg_sq


def compute_importance_scores(weights, gradients, exp_avg_sq, importance_method, alpha=0.5):
    """
    根据指定方法计算重要性得分。
    
    参数:
        weights: 权重字典
        gradients: 梯度字典
        exp_avg_sq: Adam 二阶矩字典
        importance_method: 'magnitude', 'first-order', 'second-order'
        alpha: 二阶项权重
    
    返回:
        重要性得分字典
    """
    if importance_method == 'magnitude':
        return compute_importance_scores_magnitude(weights, gradients, exp_avg_sq, alpha)
    elif importance_method == 'first-order':
        return compute_importance_scores_first_order(weights, gradients, exp_avg_sq, alpha)
    elif importance_method == 'second-order':
        return compute_importance_scores_abs(weights, gradients, exp_avg_sq, alpha)
    else:
        raise ValueError(f"Unknown importance method: {importance_method}")


def uniform_pruning(model, scores, prune_ratio, device):
    """
    Uniform 剪枝：对所有层施加相同的剪枝率。
    
    参数:
        model: PyTorch 模型
        scores: 重要性得分字典
        prune_ratio: 剪枝率 (0-1)
        device: 设备
    
    返回:
        (pruned_model, actual_prune_ratio) 二元组
    """
    # 收集所有得分
    all_scores = []
    for name, score in scores.items():
        all_scores.append(score.flatten())
    
    all_scores = torch.cat(all_scores)
    
    # 计算阈值（保留前 (1-prune_ratio) 的参数）
    threshold = torch.quantile(all_scores, prune_ratio)
    
    # 应用剪枝
    actual_pruned = 0
    total_params = 0
    
    model_params = dict(model.named_parameters())
    for name, score in scores.items():
        if name not in model_params:
            continue
        
        mask = (score >= threshold).float()
        param = model_params[name]
        param.data.mul_(mask.to(device))
        
        actual_pruned += (mask == 0).sum().item()
        total_params += mask.numel()
    
    actual_ratio = actual_pruned / total_params if total_params > 0 else 0
    return model, actual_ratio


def fit_gamma_distribution(data):
    """
    拟合 Gamma 分布。
    
    参数:
        data: numpy 数组或 torch tensor
    
    返回:
        (k, theta) 或 (None, None) 如果拟合失败
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    data_positive = data[data > 0]
    if len(data_positive) == 0:
        return None, None
    
    try:
        k, loc, scale = stats.gamma.fit(data_positive, floc=0)
        return k, scale
    except:
        return None, None


def solve_global_threshold(layer_info, global_prune_ratio):
    """
    求解全局阈值 τ*，使得全局剪枝率 = global_prune_ratio。
    
    参数:
        layer_info: 层信息列表，每个元素包含 {'name', 'n_params', 'k', 'theta', 'scores'}
        global_prune_ratio: 全局剪枝率 (0-1)
    
    返回:
        tau_star: 全局阈值
    """
    N = sum(info['n_params'] for info in layer_info)

    def objective(tau):
        """目标函数: 当前阈值下的全局剪枝率 - 目标剪枝率"""
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
            scores_np = info['scores'].cpu().numpy() if isinstance(info['scores'], torch.Tensor) else info['scores']
            all_scores.extend(scores_np.flatten().tolist())

    all_scores = np.array(all_scores)
    tau_min = np.percentile(all_scores, 0.1)
    tau_max = np.percentile(all_scores, 50)

    # 使用二分法求解
    try:
        tau_star = bisect(objective, tau_min, tau_max, xtol=1e-10, maxiter=100)
    except ValueError:
        # 如果二分法失败，尝试更大的范围
        tau_max = np.percentile(all_scores, 90)
        try:
            tau_star = bisect(objective, tau_min, tau_max, xtol=1e-10, maxiter=100)
        except ValueError:
            tau_max = np.percentile(all_scores, 99)
            tau_star = bisect(objective, tau_min, tau_max, xtol=1e-10, maxiter=100)

    return tau_star


def gamma_adaptive_pruning(model, scores, prune_ratio, device):
    """
    Gamma 自适应剪枝：拟合 Gamma 分布，求全局阈值。
    
    参数:
        model: PyTorch 模型
        scores: 重要性得分字典
        prune_ratio: 全局剪枝率 (0-1)
        device: 设备
    
    返回:
        (pruned_model, actual_prune_ratio) 二元组
    """
    # 为每层拟合 Gamma 分布
    layer_info = []
    for name, score in scores.items():
        score_np = score.cpu().numpy() if isinstance(score, torch.Tensor) else score
        k, theta = fit_gamma_distribution(score_np)
        
        layer_info.append({
            'name': name,
            'n_params': score.numel(),
            'k': k,
            'theta': theta,
            'scores': score
        })
    
    # 求解全局阈值
    tau_star = solve_global_threshold(layer_info, prune_ratio)
    
    # 应用剪枝
    actual_pruned = 0
    total_params = 0
    
    model_params = dict(model.named_parameters())
    for info in layer_info:
        name = info['name']
        score = info['scores']
        
        if name not in model_params:
            continue
        
        mask = (score >= tau_star).float()
        param = model_params[name]
        param.data.mul_(mask.to(device))
        
        actual_pruned += (mask == 0).sum().item()
        total_params += mask.numel()
    
    actual_ratio = actual_pruned / total_params if total_params > 0 else 0
    return model, actual_ratio


def run_experiment(args):
    """
    运行主实验。
    
    参数:
        args: 命令行参数
    """
    print("=" * 80)
    print("统一实验框架：论文 Table 1 核心实验")
    print("=" * 80)
    print(f"模型: {args.model}")
    print(f"重要性方法: {args.importance}")
    print(f"分配策略: {args.allocation}")
    print(f"剪枝率: {args.prune_ratios}")
    print(f"设备: {args.device}")
    print("=" * 80)
    
    # 1. 加载模型
    print("\n[1/7] 加载模型...")
    if args.model == 'gpt2-small':
        model = get_gpt2_small(pretrained=False)
    elif args.model == 'gpt2-medium':
        model = get_gpt2_medium(pretrained=False)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # 加载检查点
    if args.checkpoint:
        print(f"加载检查点: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    # 2. 加载数据
    print("\n[2/7] 加载数据...")
    train_loader = get_wikitext103_dataloader(
        split='train',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0,
        shuffle=True
    )
    
    val_loader = get_wikitext103_dataloader(
        split='validation',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0,
        shuffle=False
    )
    
    # 3. 缓存批次
    print("\n[3/7] 缓存批次...")
    print(f"缓存 {args.num_steps} 个训练批次...")
    cached_train_batches = cache_batches(train_loader, args.num_steps)
    print(f"缓存 {args.eval_batches} 个验证批次...")
    cached_eval_batches = cache_batches(val_loader, args.eval_batches)
    
    # 4. 累积梯度和动量
    print("\n[4/7] 累积梯度和动量...")
    weights, gradients, exp_avg_sq = collect_gradients_and_momentum(
        model, cached_train_batches, args.device, args.num_steps, args.model
    )
    
    # 5. 过滤可剪枝参数
    print("\n[5/7] 过滤可剪枝参数...")
    weights, gradients, exp_avg_sq = filter_prunable_params(weights, gradients, exp_avg_sq)
    print(f"可剪枝参数数量: {sum(w.numel() for w in weights.values())}")
    
    # 6. 计算重要性得分
    print(f"\n[6/7] 计算重要性得分 ({args.importance})...")
    scores = compute_importance_scores(weights, gradients, exp_avg_sq, args.importance, args.alpha)
    
    # 7. 评估原始模型
    print("\n[7/7] 评估原始模型...")
    model_original = copy.deepcopy(model)
    loss_original = evaluate_loss_with_cached_batches(
        model_original, cached_eval_batches, args.device, args.model
    )
    print(f"原始损失: {loss_original:.4f}")
    
    # 8. 对每个剪枝率进行实验
    print("\n" + "=" * 80)
    print("开始剪枝实验")
    print("=" * 80)
    
    results = []
    prune_ratios = [float(r) for r in args.prune_ratios.split(',')]
    
    for prune_ratio in prune_ratios:
        print(f"\n--- 剪枝率: {prune_ratio:.1%} ---")
        
        # 复制模型
        model_pruned = copy.deepcopy(model)
        model_pruned = model_pruned.to(args.device)
        
        # 应用剪枝
        if args.allocation == 'uniform':
            print("应用 Uniform 剪枝...")
            model_pruned, actual_ratio = uniform_pruning(
                model_pruned, scores, prune_ratio, args.device
            )
        elif args.allocation == 'gamma-adaptive':
            print("应用 Gamma 自适应剪枝...")
            model_pruned, actual_ratio = gamma_adaptive_pruning(
                model_pruned, scores, prune_ratio, args.device
            )
        else:
            raise ValueError(f"Unknown allocation: {args.allocation}")
        
        print(f"实际剪枝率: {actual_ratio:.4%}")
        
        # 评估剪枝后的损失
        loss_pruned = evaluate_loss_with_cached_batches(
            model_pruned, cached_eval_batches, args.device, args.model
        )
        
        loss_increase = loss_pruned - loss_original
        loss_increase_pct = (loss_increase / loss_original) * 100
        
        print(f"剪枝后损失: {loss_pruned:.4f}")
        print(f"损失增加: {loss_increase:.4f} ({loss_increase_pct:.2f}%)")
        
        # 记录结果
        results.append({
            'method': args.importance,
            'allocation': args.allocation,
            'prune_ratio': prune_ratio,
            'actual_ratio': actual_ratio,
            'loss_original': loss_original,
            'loss_pruned': loss_pruned,
            'loss_increase': loss_increase,
            'loss_increase_pct': loss_increase_pct
        })
    
    # 9. 保存结果
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{args.model}_{args.importance}_{args.allocation}_{timestamp}"
    
    # 保存 CSV
    csv_path = output_dir / f"{filename_base}.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"CSV 结果已保存: {csv_path}")
    
    # 保存 JSON 摘要
    json_path = output_dir / f"{filename_base}.json"
    summary = {
        'experiment': 'table1_main_experiment',
        'timestamp': timestamp,
        'config': {
            'model': args.model,
            'importance': args.importance,
            'allocation': args.allocation,
            'prune_ratios': prune_ratios,
            'num_steps': args.num_steps,
            'eval_batches': args.eval_batches,
            'batch_size': args.batch_size,
            'seq_length': args.seq_length,
            'alpha': args.alpha,
            'device': args.device,
            'checkpoint': args.checkpoint
        },
        'results': results
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"JSON 摘要已保存: {json_path}")
    
    # 打印结果表格
    print("\n" + "=" * 80)
    print("实验结果")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    print("\n实验完成！")


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description='统一实验框架：论文 Table 1 核心实验')
    
    # 模型和检查点
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='检查点路径')
    parser.add_argument('--model', type=str, choices=['gpt2-small', 'gpt2-medium'],
                        default='gpt2-small', help='模型类型')
    
    # 重要性和分配
    parser.add_argument('--importance', type=str,
                        choices=['magnitude', 'first-order', 'second-order'],
                        default='second-order', help='重要性方法')
    parser.add_argument('--allocation', type=str,
                        choices=['uniform', 'gamma-adaptive'],
                        default='gamma-adaptive', help='分配策略')
    
    # 剪枝参数
    parser.add_argument('--prune_ratios', type=str,
                        default='0.1,0.2,0.3,0.5,0.7,0.9',
                        help='逗号分隔的剪枝率')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='二阶项权重')
    
    # 训练参数
    parser.add_argument('--num_steps', type=int, default=100,
                        help='梯度累积步数')
    parser.add_argument('--eval_batches', type=int, default=10,
                        help='评估批次数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='序列长度')
    
    # 设备和输出
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--output_dir', type=str,
                        default='results/paper_results/table1',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 运行实验
    run_experiment(args)


if __name__ == '__main__':
    main()
