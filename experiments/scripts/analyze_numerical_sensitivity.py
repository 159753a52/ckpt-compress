"""
数值敏感性分析脚本。

收集训练/微调过程中的关键数值统计：
1. 梯度 (g) 的数值范围
2. 一阶动量 (m = exp_avg) 的数值范围
3. 二阶动量 (v = exp_avg_sq) 的数值范围
4. 三种参数重要性得分的数值范围
5. 参数数量统计
6. 参数重要性总和与 loss 的关系

用法:
    python experiments/scripts/analyze_numerical_sensitivity.py --max_samples 100
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, GPT2Config
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.ckpt_compress.utils.data_loader import get_wikitext2_dataloader
from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.methods.adam_prune.importance import (
    compute_importance_scores,
    compute_importance_scores_abs,
    compute_importance_scores_first_order,
)


@dataclass
class TensorStats:
    """张量统计信息"""
    name: str
    shape: Tuple
    numel: int
    min: float
    max: float
    mean: float
    std: float
    abs_mean: float
    abs_max: float
    nonzero_ratio: float

    def __str__(self):
        return (
            f"{self.name}: shape={self.shape}, numel={self.numel:,}\n"
            f"  range: [{self.min:.2e}, {self.max:.2e}]\n"
            f"  mean={self.mean:.2e}, std={self.std:.2e}\n"
            f"  |mean|={self.abs_mean:.2e}, |max|={self.abs_max:.2e}\n"
            f"  nonzero={self.nonzero_ratio:.2%}"
        )


def compute_tensor_stats(tensor: torch.Tensor, name: str) -> TensorStats:
    """计算张量的统计信息"""
    t = tensor.float().detach()
    return TensorStats(
        name=name,
        shape=tuple(tensor.shape),
        numel=tensor.numel(),
        min=t.min().item(),
        max=t.max().item(),
        mean=t.mean().item(),
        std=t.std().item(),
        abs_mean=t.abs().mean().item(),
        abs_max=t.abs().max().item(),
        nonzero_ratio=(t != 0).float().mean().item(),
    )


def aggregate_stats(stats_list: list) -> Dict:
    """聚合多个张量的统计信息"""
    total_numel = sum(s.numel for s in stats_list)

    # 加权平均
    weighted_mean = sum(s.mean * s.numel for s in stats_list) / total_numel
    weighted_abs_mean = sum(s.abs_mean * s.numel for s in stats_list) / total_numel

    return {
        "total_params": total_numel,
        "num_tensors": len(stats_list),
        "global_min": min(s.min for s in stats_list),
        "global_max": max(s.max for s in stats_list),
        "global_abs_max": max(s.abs_max for s in stats_list),
        "weighted_mean": weighted_mean,
        "weighted_abs_mean": weighted_abs_mean,
    }


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def analyze_model(
    model: nn.Module,
    optimizer: torch.optim.Adam,
    data_batch: Dict[str, torch.Tensor],
    device: str = "cuda",
) -> Dict:
    """
    分析模型的数值敏感性。

    返回包含所有统计信息的字典。
    """
    results = {}

    # ========== 1. 参数统计 ==========
    print_section("1. 模型参数 (θ)")

    weights = {}
    weight_stats = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights[name] = param.data
            stats = compute_tensor_stats(param.data, name)
            weight_stats.append(stats)

    # 打印前5层和聚合统计
    for s in weight_stats[:5]:
        print(s)
    print(f"... 共 {len(weight_stats)} 个参数张量")

    agg = aggregate_stats(weight_stats)
    print(f"\n[聚合统计]")
    print(f"  总参数量: {agg['total_params']:,}")
    print(f"  全局范围: [{agg['global_min']:.2e}, {agg['global_max']:.2e}]")
    print(f"  加权均值: {agg['weighted_mean']:.2e}")
    print(f"  加权|均值|: {agg['weighted_abs_mean']:.2e}")

    results["weights"] = agg

    # ========== 2. 计算梯度 ==========
    print_section("2. 梯度 (g = ∂L/∂θ)")

    model.zero_grad()
    input_ids = data_batch["input_ids"]
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()

    print(f"当前 Loss: {loss.item():.4f}")
    results["loss"] = loss.item()

    gradients = {}
    grad_stats = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradients[name] = param.grad
            stats = compute_tensor_stats(param.grad, name)
            grad_stats.append(stats)

    for s in grad_stats[:5]:
        print(s)
    print(f"... 共 {len(grad_stats)} 个梯度张量")

    agg = aggregate_stats(grad_stats)
    print(f"\n[聚合统计]")
    print(f"  全局范围: [{agg['global_min']:.2e}, {agg['global_max']:.2e}]")
    print(f"  加权均值: {agg['weighted_mean']:.2e}")
    print(f"  加权|均值|: {agg['weighted_abs_mean']:.2e}")

    results["gradients"] = agg

    # ========== 3. Adam 一阶动量 (m = exp_avg) ==========
    print_section("3. 一阶动量 (m = exp_avg)")

    exp_avg = {}
    exp_avg_stats = []

    # 从 optimizer state 中提取
    for name, param in model.named_parameters():
        if param.requires_grad:
            state = optimizer.state.get(param, {})
            if "exp_avg" in state:
                exp_avg[name] = state["exp_avg"]
                stats = compute_tensor_stats(state["exp_avg"], name)
                exp_avg_stats.append(stats)

    if exp_avg_stats:
        for s in exp_avg_stats[:5]:
            print(s)
        print(f"... 共 {len(exp_avg_stats)} 个一阶动量张量")

        agg = aggregate_stats(exp_avg_stats)
        print(f"\n[聚合统计]")
        print(f"  全局范围: [{agg['global_min']:.2e}, {agg['global_max']:.2e}]")
        print(f"  加权均值: {agg['weighted_mean']:.2e}")
        print(f"  加权|均值|: {agg['weighted_abs_mean']:.2e}")
        results["exp_avg"] = agg
    else:
        print("  (优化器状态为空，需要先执行几步训练)")
        results["exp_avg"] = None

    # ========== 4. Adam 二阶动量 (v = exp_avg_sq) ==========
    print_section("4. 二阶动量 (v = exp_avg_sq)")

    exp_avg_sq = {}
    exp_avg_sq_stats = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            state = optimizer.state.get(param, {})
            if "exp_avg_sq" in state:
                exp_avg_sq[name] = state["exp_avg_sq"]
                stats = compute_tensor_stats(state["exp_avg_sq"], name)
                exp_avg_sq_stats.append(stats)

    if exp_avg_sq_stats:
        for s in exp_avg_sq_stats[:5]:
            print(s)
        print(f"... 共 {len(exp_avg_sq_stats)} 个二阶动量张量")

        agg = aggregate_stats(exp_avg_sq_stats)
        print(f"\n[聚合统计]")
        print(f"  全局范围: [{agg['global_min']:.2e}, {agg['global_max']:.2e}]")
        print(f"  加权均值: {agg['weighted_mean']:.2e}")
        print(f"  加权|均值|: {agg['weighted_abs_mean']:.2e}")
        results["exp_avg_sq"] = agg
    else:
        print("  (优化器状态为空，需要先执行几步训练)")
        results["exp_avg_sq"] = None

    # ========== 5. 参数重要性得分 ==========
    if exp_avg_sq:
        print_section("5. 参数重要性得分")

        # 5.1 Adam 近似 (保留符号)
        print("\n--- 5.1 Adam近似 (s = -g·θ + α·v·θ²) ---")
        scores_adam = compute_importance_scores(weights, gradients, exp_avg_sq, alpha=0.5)

        adam_stats = []
        for name, score in scores_adam.items():
            stats = compute_tensor_stats(score, name)
            adam_stats.append(stats)

        for s in adam_stats[:3]:
            print(s)

        agg = aggregate_stats(adam_stats)
        print(f"\n[聚合统计]")
        print(f"  全局范围: [{agg['global_min']:.2e}, {agg['global_max']:.2e}]")
        print(f"  加权均值: {agg['weighted_mean']:.2e}")
        print(f"  加权|均值|: {agg['weighted_abs_mean']:.2e}")
        results["importance_adam"] = agg

        # 计算重要性总和
        total_importance = sum(s.data.sum().item() for s in scores_adam.values())
        print(f"\n  重要性总和 Σs_i = {total_importance:.4e}")
        results["total_importance_adam"] = total_importance

        # 5.2 绝对值版本
        print("\n--- 5.2 绝对值版本 (d = |g·θ| + α·|v·θ²|) ---")
        scores_abs = compute_importance_scores_abs(weights, gradients, exp_avg_sq, alpha=0.5)

        abs_stats = []
        for name, score in scores_abs.items():
            stats = compute_tensor_stats(score, name)
            abs_stats.append(stats)

        agg = aggregate_stats(abs_stats)
        print(f"[聚合统计]")
        print(f"  全局范围: [{agg['global_min']:.2e}, {agg['global_max']:.2e}]")
        print(f"  加权均值: {agg['weighted_mean']:.2e}")

        total_importance_abs = sum(s.data.sum().item() for s in scores_abs.values())
        print(f"  重要性总和 Σd_i = {total_importance_abs:.4e}")
        results["importance_abs"] = agg
        results["total_importance_abs"] = total_importance_abs

        # 5.3 仅一阶项
        print("\n--- 5.3 仅一阶项 (d = |g·θ|) ---")
        scores_first = compute_importance_scores_first_order(weights, gradients, exp_avg_sq)

        first_stats = []
        for name, score in scores_first.items():
            stats = compute_tensor_stats(score, name)
            first_stats.append(stats)

        agg = aggregate_stats(first_stats)
        print(f"[聚合统计]")
        print(f"  全局范围: [{agg['global_min']:.2e}, {agg['global_max']:.2e}]")
        print(f"  加权均值: {agg['weighted_mean']:.2e}")

        total_importance_first = sum(s.data.sum().item() for s in scores_first.values())
        print(f"  重要性总和 Σd_i = {total_importance_first:.4e}")
        results["importance_first_order"] = agg
        results["total_importance_first"] = total_importance_first

    # ========== 6. 重要性与 Loss 的关系 ==========
    print_section("6. 重要性与 Loss 的关系分析")

    print(f"\n当前 Loss: {results['loss']:.4f}")

    if exp_avg_sq:
        print(f"\n理论分析:")
        print(f"  泰勒展开: ΔL ≈ -g^T·θ + 0.5·θ^T·H·θ")
        print(f"  当删除所有参数时 (Δθ = -θ):")
        print(f"    预测 ΔL = Σs_i = {results['total_importance_adam']:.4e}")
        print(f"\n  如果 Σs_i ≈ L，说明重要性得分可以很好地预测 loss")
        print(f"  当前比值: Σs_i / L = {results['total_importance_adam'] / results['loss']:.4f}")

        # 分析一阶项和二阶项的贡献
        first_order_total = sum(
            (-gradients[name] * weights[name]).sum().item()
            for name in weights if name in gradients
        )
        second_order_total = sum(
            (0.5 * exp_avg_sq[name] * weights[name] ** 2).sum().item()
            for name in weights if name in exp_avg_sq
        )

        print(f"\n  一阶项总和 (-g^T·θ): {first_order_total:.4e}")
        print(f"  二阶项总和 (0.5·θ^T·H·θ): {second_order_total:.4e}")
        print(f"  二阶/一阶比值: {abs(second_order_total / first_order_total) if first_order_total != 0 else float('inf'):.4f}")

        results["first_order_total"] = first_order_total
        results["second_order_total"] = second_order_total

    return results


def run_training_steps(
    model: nn.Module,
    optimizer: torch.optim.Adam,
    dataloader,
    num_steps: int = 10,
    device: str = "cuda",
) -> list:
    """执行几步训练以初始化优化器状态"""
    model.train()
    losses = []

    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"  Step {i+1}/{num_steps}, Loss: {loss.item():.4f}")

    return losses


def main():
    parser = argparse.ArgumentParser(description="数值敏感性分析")
    parser.add_argument("--model", type=str, default="gpt2", help="模型名称")
    parser.add_argument("--max_samples", type=int, default=100, help="最大样本数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--warmup_steps", type=int, default=10, help="预热步数")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"设备: {args.device}")
    print(f"模型: {args.model}")

    # 加载模型（使用随机初始化，避免网络问题）
    print("\n加载模型...")
    if args.model == "gpt2":
        # 使用随机初始化的 GPT-2 small
        wrapper = get_gpt2_small(pretrained=False)
        model = wrapper.model
    else:
        # 尝试从本地缓存加载
        cache_dir = "data/models"
        try:
            model = GPT2LMHeadModel.from_pretrained(args.model, cache_dir=cache_dir, local_files_only=True)
        except Exception:
            print(f"无法加载 {args.model}，使用随机初始化的 GPT-2 small")
            wrapper = get_gpt2_small(pretrained=False)
            model = wrapper.model
    model.to(args.device)

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
        seq_length=128,
        max_samples=args.max_samples,
    )

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 预热训练（初始化优化器状态）
    print(f"\n执行 {args.warmup_steps} 步预热训练...")
    warmup_losses = run_training_steps(
        model, optimizer, dataloader,
        num_steps=args.warmup_steps,
        device=args.device
    )

    # 获取一个新的 batch 用于分析
    print("\n获取分析用数据批次...")
    dataloader = get_wikitext2_dataloader(
        split="train",
        batch_size=args.batch_size,
        seq_length=128,
        max_samples=args.max_samples,
    )
    batch = next(iter(dataloader))
    batch = {k: v.to(args.device) for k, v in batch.items()}

    # 执行分析
    print("\n" + "=" * 60)
    print(" 开始数值敏感性分析")
    print("=" * 60)

    results = analyze_model(model, optimizer, batch, device=args.device)

    # 总结
    print_section("总结")
    print(f"""
数值范围总结:
┌─────────────────┬──────────────────┬──────────────────┐
│ 类型            │ 典型范围         │ |均值|           │
├─────────────────┼──────────────────┼──────────────────┤
│ 参数 θ          │ [{results['weights']['global_min']:.1e}, {results['weights']['global_max']:.1e}] │ {results['weights']['weighted_abs_mean']:.2e}       │
│ 梯度 g          │ [{results['gradients']['global_min']:.1e}, {results['gradients']['global_max']:.1e}] │ {results['gradients']['weighted_abs_mean']:.2e}       │
│ 一阶动量 m      │ [{results['exp_avg']['global_min']:.1e}, {results['exp_avg']['global_max']:.1e}] │ {results['exp_avg']['weighted_abs_mean']:.2e}       │
│ 二阶动量 v      │ [{results['exp_avg_sq']['global_min']:.1e}, {results['exp_avg_sq']['global_max']:.1e}] │ {results['exp_avg_sq']['weighted_abs_mean']:.2e}       │
└─────────────────┴──────────────────┴──────────────────┘

参数重要性:
┌─────────────────┬──────────────────┬──────────────────┐
│ 方法            │ 范围             │ 总和 Σ           │
├─────────────────┼──────────────────┼──────────────────┤
│ Adam近似        │ [{results['importance_adam']['global_min']:.1e}, {results['importance_adam']['global_max']:.1e}] │ {results['total_importance_adam']:.2e}       │
│ 绝对值版本      │ [{results['importance_abs']['global_min']:.1e}, {results['importance_abs']['global_max']:.1e}] │ {results['total_importance_abs']:.2e}       │
│ 仅一阶项        │ [{results['importance_first_order']['global_min']:.1e}, {results['importance_first_order']['global_max']:.1e}] │ {results['total_importance_first']:.2e}       │
└─────────────────┴──────────────────┴──────────────────┘

Loss 关系:
  当前 Loss: {results['loss']:.4f}
  Σs_i / Loss = {results['total_importance_adam'] / results['loss']:.4f}

  一阶项贡献: {results['first_order_total']:.4e}
  二阶项贡献: {results['second_order_total']:.4e}
""")


if __name__ == "__main__":
    main()
