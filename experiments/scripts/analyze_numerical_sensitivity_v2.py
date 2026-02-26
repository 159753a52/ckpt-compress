"""
数值敏感性分析脚本 v2 - 从检查点加载并继续训练。

从微调后的检查点加载，继续训练并记录数值变化：
1. 梯度 (g) 的数值范围
2. 一阶动量 (m = exp_avg) 的数值范围
3. 二阶动量 (v = exp_avg_sq) 的数值范围
4. 三种参数重要性得分的数值范围
5. 参数重要性总和与 loss 的关系

用法:
    python experiments/scripts/analyze_numerical_sensitivity_v2.py \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --train_steps 100
"""

import argparse
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from dataclasses import dataclass, asdict
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.ckpt_compress.utils.data_loader import get_wikitext2_dataloader
from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.methods.adam_prune.importance import (
    compute_importance_scores,
    compute_importance_scores_abs,
    compute_importance_scores_first_order,
)


@dataclass
class StepStats:
    """单步统计信息"""
    step: int
    loss: float
    # 参数统计
    weight_abs_mean: float
    weight_abs_max: float
    # 梯度统计
    grad_abs_mean: float
    grad_abs_max: float
    # 一阶动量统计
    exp_avg_abs_mean: float
    exp_avg_abs_max: float
    # 二阶动量统计
    exp_avg_sq_mean: float
    exp_avg_sq_max: float
    # 重要性得分统计
    importance_adam_sum: float
    importance_adam_abs_mean: float
    importance_abs_sum: float
    importance_first_order_sum: float
    # 一阶/二阶项分解
    first_order_term: float
    second_order_term: float
    # 比值
    importance_loss_ratio: float


def compute_aggregated_stats(tensors: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    """计算张量字典的聚合统计（加权|均值|和最大|值|）"""
    total_numel = sum(t.numel() for t in tensors.values())
    weighted_abs_mean = sum(
        t.abs().mean().item() * t.numel() for t in tensors.values()
    ) / total_numel
    abs_max = max(t.abs().max().item() for t in tensors.values())
    return weighted_abs_mean, abs_max


def analyze_step(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    batch: Dict[str, torch.Tensor],
    step: int,
) -> StepStats:
    """分析单步的数值统计"""

    # 收集权重
    weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights[name] = param.data

    weight_abs_mean, weight_abs_max = compute_aggregated_stats(weights)

    # 计算梯度
    model.zero_grad()
    input_ids = batch["input_ids"]
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()

    # 收集梯度
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradients[name] = param.grad

    grad_abs_mean, grad_abs_max = compute_aggregated_stats(gradients)

    # 收集优化器状态
    exp_avg = {}
    exp_avg_sq = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state = optimizer.state.get(param, {})
            if "exp_avg" in state:
                exp_avg[name] = state["exp_avg"]
            if "exp_avg_sq" in state:
                exp_avg_sq[name] = state["exp_avg_sq"]

    if exp_avg:
        exp_avg_abs_mean, exp_avg_abs_max = compute_aggregated_stats(exp_avg)
    else:
        exp_avg_abs_mean, exp_avg_abs_max = 0.0, 0.0

    if exp_avg_sq:
        # 二阶动量是非负的，直接用 mean 和 max
        total_numel = sum(t.numel() for t in exp_avg_sq.values())
        exp_avg_sq_mean = sum(
            t.mean().item() * t.numel() for t in exp_avg_sq.values()
        ) / total_numel
        exp_avg_sq_max = max(t.max().item() for t in exp_avg_sq.values())
    else:
        exp_avg_sq_mean, exp_avg_sq_max = 0.0, 0.0

    # 计算重要性得分
    if exp_avg_sq:
        # Adam 近似
        scores_adam = compute_importance_scores(weights, gradients, exp_avg_sq, alpha=0.5)
        importance_adam_sum = sum(s.sum().item() for s in scores_adam.values())
        importance_adam_abs_mean, _ = compute_aggregated_stats(scores_adam)

        # 绝对值版本
        scores_abs = compute_importance_scores_abs(weights, gradients, exp_avg_sq, alpha=0.5)
        importance_abs_sum = sum(s.sum().item() for s in scores_abs.values())

        # 仅一阶项
        scores_first = compute_importance_scores_first_order(weights, gradients, exp_avg_sq)
        importance_first_order_sum = sum(s.sum().item() for s in scores_first.values())

        # 一阶/二阶项分解
        first_order_term = sum(
            (-gradients[name] * weights[name]).sum().item()
            for name in weights if name in gradients
        )
        second_order_term = sum(
            (0.5 * exp_avg_sq[name] * weights[name] ** 2).sum().item()
            for name in weights if name in exp_avg_sq
        )
    else:
        importance_adam_sum = 0.0
        importance_adam_abs_mean = 0.0
        importance_abs_sum = 0.0
        importance_first_order_sum = 0.0
        first_order_term = 0.0
        second_order_term = 0.0

    # 计算比值
    loss_val = loss.item()
    importance_loss_ratio = importance_adam_sum / loss_val if loss_val > 0 else 0.0

    return StepStats(
        step=step,
        loss=loss_val,
        weight_abs_mean=weight_abs_mean,
        weight_abs_max=weight_abs_max,
        grad_abs_mean=grad_abs_mean,
        grad_abs_max=grad_abs_max,
        exp_avg_abs_mean=exp_avg_abs_mean,
        exp_avg_abs_max=exp_avg_abs_max,
        exp_avg_sq_mean=exp_avg_sq_mean,
        exp_avg_sq_max=exp_avg_sq_max,
        importance_adam_sum=importance_adam_sum,
        importance_adam_abs_mean=importance_adam_abs_mean,
        importance_abs_sum=importance_abs_sum,
        importance_first_order_sum=importance_first_order_sum,
        first_order_term=first_order_term,
        second_order_term=second_order_term,
        importance_loss_ratio=importance_loss_ratio,
    )


def print_step_stats(stats: StepStats):
    """打印单步统计"""
    print(f"\n{'='*60}")
    print(f" Step {stats.step} | Loss: {stats.loss:.4f}")
    print(f"{'='*60}")
    print(f"  参数 θ:     |mean|={stats.weight_abs_mean:.2e}, |max|={stats.weight_abs_max:.2e}")
    print(f"  梯度 g:     |mean|={stats.grad_abs_mean:.2e}, |max|={stats.grad_abs_max:.2e}")
    print(f"  一阶动量 m: |mean|={stats.exp_avg_abs_mean:.2e}, |max|={stats.exp_avg_abs_max:.2e}")
    print(f"  二阶动量 v: mean={stats.exp_avg_sq_mean:.2e}, max={stats.exp_avg_sq_max:.2e}")
    print(f"  ---")
    print(f"  重要性 Σs_i (Adam):    {stats.importance_adam_sum:.4e}")
    print(f"  重要性 Σd_i (绝对值):  {stats.importance_abs_sum:.4e}")
    print(f"  重要性 Σd_i (一阶):    {stats.importance_first_order_sum:.4e}")
    print(f"  ---")
    print(f"  一阶项 (-g^T·θ):       {stats.first_order_term:.4e}")
    print(f"  二阶项 (0.5·θ^T·H·θ): {stats.second_order_term:.4e}")
    print(f"  Σs_i / Loss:           {stats.importance_loss_ratio:.4f}")


def main():
    parser = argparse.ArgumentParser(description="数值敏感性分析 v2")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt",
        help="检查点路径"
    )
    parser.add_argument("--train_steps", type=int, default=100, help="继续训练步数")
    parser.add_argument("--log_interval", type=int, default=10, help="记录间隔")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--seq_length", type=int, default=512, help="序列长度")
    parser.add_argument("--max_samples", type=int, default=500, help="最大样本数")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="results/numerical_sensitivity.json", help="输出文件")
    args = parser.parse_args()

    print(f"设备: {args.device}")
    print(f"检查点: {args.checkpoint}")

    # 加载模型
    print("\n加载模型...")
    wrapper = get_gpt2_small(pretrained=False)
    model = wrapper.model

    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 加载检查点
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = checkpoint.get("step", 1000)
    print(f"从 step {start_step} 继续训练")

    model.to(args.device)

    # 将优化器状态移到正确的设备
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(args.device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 加载数据
    print("\n加载数据...")
    dataloader = get_wikitext2_dataloader(
        split="train",
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        max_samples=args.max_samples,
    )

    # 训练并记录
    print(f"\n开始训练 {args.train_steps} 步...")
    model.train()

    all_stats: List[StepStats] = []
    data_iter = iter(dataloader)

    for i in range(args.train_steps):
        current_step = start_step + i + 1

        # 获取数据
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(args.device) for k, v in batch.items()}

        # 记录统计（在优化器更新前）
        if i % args.log_interval == 0 or i == args.train_steps - 1:
            stats = analyze_step(model, optimizer, batch, current_step)
            all_stats.append(stats)
            print_step_stats(stats)

        # 训练步骤
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # 打印总结
    print("\n" + "=" * 60)
    print(" 训练过程数值变化总结")
    print("=" * 60)

    print(f"\n{'Step':>6} | {'Loss':>8} | {'|g|_mean':>10} | {'v_mean':>10} | {'Σs_i':>12} | {'Σs_i/L':>8}")
    print("-" * 70)
    for s in all_stats:
        print(f"{s.step:>6} | {s.loss:>8.4f} | {s.grad_abs_mean:>10.2e} | {s.exp_avg_sq_mean:>10.2e} | {s.importance_adam_sum:>12.4e} | {s.importance_loss_ratio:>8.4f}")

    # 分析变化趋势
    if len(all_stats) >= 2:
        first = all_stats[0]
        last = all_stats[-1]

        print(f"\n变化趋势 (step {first.step} → {last.step}):")
        print(f"  Loss:           {first.loss:.4f} → {last.loss:.4f} ({(last.loss - first.loss) / first.loss * 100:+.2f}%)")
        print(f"  |g|_mean:       {first.grad_abs_mean:.2e} → {last.grad_abs_mean:.2e}")
        print(f"  v_mean:         {first.exp_avg_sq_mean:.2e} → {last.exp_avg_sq_mean:.2e}")
        print(f"  Σs_i:           {first.importance_adam_sum:.4e} → {last.importance_adam_sum:.4e}")
        print(f"  Σs_i/Loss:      {first.importance_loss_ratio:.4f} → {last.importance_loss_ratio:.4f}")
        print(f"  一阶项:         {first.first_order_term:.4e} → {last.first_order_term:.4e}")
        print(f"  二阶项:         {first.second_order_term:.4e} → {last.second_order_term:.4e}")

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "start_step": start_step,
            "train_steps": args.train_steps,
            "total_params": total_params,
        },
        "stats": [asdict(s) for s in all_stats],
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
