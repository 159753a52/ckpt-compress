"""
基于权重绝对值的重要性计算示例。

展示如何使用 compute_importance_scores_magnitude 方法进行参数重要性评估。
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ckpt_compress.models.gpt2 import get_gpt2_medium
from src.ckpt_compress.methods.adam_prune.importance import (
    compute_importance_scores_magnitude,
    compute_importance_scores,
    compute_importance_scores_first_order,
    get_flattened_scores,
)


def example_1_basic_usage():
    """示例 1: 基本使用"""
    print("\n" + "=" * 60)
    print("示例 1: 基于权重绝对值的重要性计算")
    print("=" * 60)

    # 创建简单的权重
    weights = {
        'layer1.weight': torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
        'layer1.bias': torch.tensor([0.5, -0.5]),
    }

    # 计算重要性得分（仅基于权重绝对值）
    scores = compute_importance_scores_magnitude(weights)

    print("\n权重:")
    for name, weight in weights.items():
        print(f"  {name}: {weight}")

    print("\n重要性得分（权重绝对值）:")
    for name, score in scores.items():
        print(f"  {name}: {score}")

    print("\n✓ 优点: 不需要梯度，计算极快")


def example_2_compare_methods():
    """示例 2: 对比不同的重要性计算方法"""
    print("\n" + "=" * 60)
    print("示例 2: 对比不同的重要性计算方法")
    print("=" * 60)

    # 准备数据
    weights = {
        'layer1': torch.tensor([1.0, -2.0, 3.0, -4.0, 0.5]),
    }
    gradients = {
        'layer1': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.05]),
    }
    exp_avg_sq = {
        'layer1': torch.tensor([0.01, 0.02, 0.03, 0.04, 0.005]),
    }

    # 方法 1: 权重绝对值（最简单）
    scores_magnitude = compute_importance_scores_magnitude(weights)

    # 方法 2: 一阶梯度 |g * θ|
    scores_first_order = compute_importance_scores_first_order(
        weights, gradients, exp_avg_sq
    )

    # 方法 3: Adam 二阶矩近似
    scores_adam = compute_importance_scores(
        weights, gradients, exp_avg_sq, alpha=0.5
    )

    print("\n权重: ", weights['layer1'])
    print("梯度: ", gradients['layer1'])

    print("\n方法 1 - 权重绝对值 |θ|:")
    print(f"  得分: {scores_magnitude['layer1']}")
    print(f"  最重要参数索引: {torch.argmax(scores_magnitude['layer1']).item()}")

    print("\n方法 2 - 一阶梯度 |g * θ|:")
    print(f"  得分: {scores_first_order['layer1']}")
    print(f"  最重要参数索引: {torch.argmax(scores_first_order['layer1']).item()}")

    print("\n方法 3 - Adam 二阶矩:")
    print(f"  得分: {scores_adam['layer1']}")
    print(f"  最重要参数索引: {torch.argmax(scores_adam['layer1']).item()}")

    print("\n观察: 不同方法可能给出不同的重要性排序")


def example_3_gpt2_model():
    """示例 3: 在 GPT-2 模型上使用"""
    print("\n" + "=" * 60)
    print("示例 3: 在 GPT-2 Medium 模型上计算权重绝对值重要性")
    print("=" * 60)

    # 加载模型
    print("加载 GPT-2 Medium 模型...")
    model = get_gpt2_medium(pretrained=True)

    # 收集权重
    weights = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    print(f"✓ 模型加载完成，共 {len(weights)} 个参数层")

    # 计算重要性得分（仅基于权重绝对值）
    print("\n计算重要性得分（基于权重绝对值）...")
    scores = compute_importance_scores_magnitude(weights)

    print(f"✓ 计算完成")

    # 分析前 5 个层的重要性
    print("\n前 5 个层的重要性统计:")
    for i, (name, score) in enumerate(list(scores.items())[:5]):
        mean_score = score.mean().item()
        max_score = score.max().item()
        min_score = score.min().item()
        print(f"  {i+1}. {name}:")
        print(f"     平均: {mean_score:.6f}, 最大: {max_score:.6f}, 最小: {min_score:.6f}")

    # 全局统计
    flat_scores = get_flattened_scores(scores)
    print(f"\n全局统计:")
    print(f"  总参数数: {flat_scores.numel():,}")
    print(f"  平均重要性: {flat_scores.mean().item():.6f}")
    print(f"  最大重要性: {flat_scores.max().item():.6f}")
    print(f"  最小重要性: {flat_scores.min().item():.6f}")

    # 计算不同阈值下的稀疏度
    print(f"\n不同阈值下的稀疏度:")
    for percentile in [50, 75, 90, 95, 99]:
        threshold = torch.quantile(flat_scores, percentile / 100.0)
        sparsity = (flat_scores < threshold).float().mean().item()
        print(f"  {percentile}% 分位数 (阈值={threshold:.6f}): 稀疏度 {sparsity*100:.2f}%")


def example_4_pruning_simulation():
    """示例 4: 模拟剪枝过程"""
    print("\n" + "=" * 60)
    print("示例 4: 基于权重绝对值的剪枝模拟")
    print("=" * 60)

    # 创建一个简单的权重矩阵
    torch.manual_seed(42)
    weights = {
        'layer1': torch.randn(10, 10),
    }

    # 计算重要性得分
    scores = compute_importance_scores_magnitude(weights)

    # 展平得分
    flat_scores = scores['layer1'].flatten()
    flat_weights = weights['layer1'].flatten()

    print(f"原始权重统计:")
    print(f"  形状: {weights['layer1'].shape}")
    print(f"  非零参数: {(flat_weights != 0).sum().item()}/{flat_weights.numel()}")
    print(f"  平均绝对值: {flat_weights.abs().mean().item():.6f}")

    # 模拟不同稀疏度的剪枝
    print(f"\n剪枝模拟:")
    for target_sparsity in [0.3, 0.5, 0.7, 0.9]:
        # 计算阈值
        threshold = torch.quantile(flat_scores, target_sparsity)

        # 创建掩码
        mask = (flat_scores >= threshold).float()

        # 应用掩码
        pruned_weights = flat_weights * mask

        # 统计
        actual_sparsity = (pruned_weights == 0).float().mean().item()
        remaining_params = (pruned_weights != 0).sum().item()

        print(f"  目标稀疏度 {target_sparsity*100:.0f}%:")
        print(f"    阈值: {threshold:.6f}")
        print(f"    实际稀疏度: {actual_sparsity*100:.2f}%")
        print(f"    剩余参数: {remaining_params}/{flat_weights.numel()}")


def example_5_layer_wise_importance():
    """示例 5: 逐层重要性分析"""
    print("\n" + "=" * 60)
    print("示例 5: 逐层重要性分析")
    print("=" * 60)

    # 创建多层权重
    torch.manual_seed(42)
    weights = {
        f'layer{i}.weight': torch.randn(20, 20) * (0.1 * (i + 1))
        for i in range(5)
    }

    # 计算重要性得分
    scores = compute_importance_scores_magnitude(weights)

    # 逐层分析
    print("\n逐层重要性统计:")
    layer_stats = []
    for name in sorted(weights.keys()):
        score = scores[name]
        stats = {
            'name': name,
            'mean': score.mean().item(),
            'std': score.std().item(),
            'max': score.max().item(),
            'min': score.min().item(),
        }
        layer_stats.append(stats)

        print(f"  {name}:")
        print(f"    平均: {stats['mean']:.6f} ± {stats['std']:.6f}")
        print(f"    范围: [{stats['min']:.6f}, {stats['max']:.6f}]")

    # 找出最重要和最不重要的层
    layer_means = [s['mean'] for s in layer_stats]
    most_important_idx = layer_means.index(max(layer_means))
    least_important_idx = layer_means.index(min(layer_means))

    print(f"\n最重要的层: {layer_stats[most_important_idx]['name']}")
    print(f"  平均重要性: {layer_stats[most_important_idx]['mean']:.6f}")

    print(f"\n最不重要的层: {layer_stats[least_important_idx]['name']}")
    print(f"  平均重要性: {layer_stats[least_important_idx]['mean']:.6f}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("基于权重绝对值的重要性计算示例")
    print("=" * 60)

    # 运行所有示例
    example_1_basic_usage()
    example_2_compare_methods()
    example_3_gpt2_model()
    example_4_pruning_simulation()
    example_5_layer_wise_importance()

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("\n权重绝对值方法 (Magnitude-based Pruning):")
    print("  ✓ 优点:")
    print("    - 计算极快，不需要梯度")
    print("    - 不需要训练数据")
    print("    - 适合快速剪枝和基线对比")
    print("  ✗ 缺点:")
    print("    - 不考虑参数对损失的实际影响")
    print("    - 可能剪掉绝对值小但重要的参数")
    print("\n适用场景:")
    print("  - 快速剪枝原型")
    print("  - 结构化剪枝（神经元、通道）")
    print("  - 作为其他方法的基线对比")
    print("\n使用方法:")
    print("  from src.ckpt_compress.methods.adam_prune.importance import \\")
    print("      compute_importance_scores_magnitude")
    print("  scores = compute_importance_scores_magnitude(weights)")


if __name__ == '__main__':
    main()
