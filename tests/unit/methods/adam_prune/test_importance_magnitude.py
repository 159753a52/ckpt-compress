"""
测试基于权重绝对值的重要性计算方法。
"""

import torch
import pytest
from src.ckpt_compress.methods.adam_prune.importance import (
    compute_importance_scores_magnitude,
    compute_importance_scores,
    compute_importance_scores_first_order,
    get_flattened_scores,
)


def test_compute_importance_scores_magnitude_basic():
    """测试基本功能。"""
    weights = {
        'layer1.weight': torch.tensor([[1.0, -2.0], [3.0, -4.0]]),
        'layer1.bias': torch.tensor([0.5, -0.5]),
    }

    scores = compute_importance_scores_magnitude(weights)

    # 验证返回的键
    assert set(scores.keys()) == set(weights.keys())

    # 验证形状
    assert scores['layer1.weight'].shape == weights['layer1.weight'].shape
    assert scores['layer1.bias'].shape == weights['layer1.bias'].shape

    # 验证值（应该是绝对值）
    expected_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_bias = torch.tensor([0.5, 0.5])

    assert torch.allclose(scores['layer1.weight'], expected_weight)
    assert torch.allclose(scores['layer1.bias'], expected_bias)


def test_compute_importance_scores_magnitude_no_gradients():
    """测试不需要梯度信息。"""
    weights = {
        'layer1.weight': torch.randn(10, 5),
    }

    # 不传入梯度和优化器状态
    scores = compute_importance_scores_magnitude(weights)

    assert 'layer1.weight' in scores
    assert scores['layer1.weight'].shape == weights['layer1.weight'].shape
    assert torch.all(scores['layer1.weight'] >= 0)  # 所有值应该非负


def test_compute_importance_scores_magnitude_all_positive():
    """测试所有得分都是非负的。"""
    weights = {
        'layer1': torch.randn(100, 50),
        'layer2': torch.randn(50, 20),
    }

    scores = compute_importance_scores_magnitude(weights)

    for name, score in scores.items():
        assert torch.all(score >= 0), f"Layer {name} has negative scores"


def test_compute_importance_scores_magnitude_vs_abs():
    """测试结果与直接取绝对值一致。"""
    weights = {
        'layer1': torch.randn(10, 5),
    }

    scores = compute_importance_scores_magnitude(weights)
    expected = torch.abs(weights['layer1'])

    assert torch.allclose(scores['layer1'], expected)


def test_compute_importance_scores_magnitude_empty():
    """测试空输入。"""
    weights = {}
    scores = compute_importance_scores_magnitude(weights)
    assert scores == {}


def test_compute_importance_scores_magnitude_interface_compatibility():
    """测试接口兼容性（可以传入但不使用的参数）。"""
    weights = {
        'layer1': torch.randn(5, 3),
    }
    gradients = {
        'layer1': torch.randn(5, 3),
    }
    exp_avg_sq = {
        'layer1': torch.randn(5, 3),
    }

    # 应该可以传入这些参数，但不会使用
    scores = compute_importance_scores_magnitude(
        weights,
        gradients=gradients,
        exp_avg_sq=exp_avg_sq,
        alpha=0.5
    )

    # 结果应该只依赖于权重
    expected = torch.abs(weights['layer1'])
    assert torch.allclose(scores['layer1'], expected)


def test_compare_magnitude_with_other_methods():
    """对比不同重要性计算方法。"""
    # 准备数据
    weights = {
        'layer1': torch.tensor([[1.0, -2.0], [3.0, -4.0]]),
    }
    gradients = {
        'layer1': torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    }
    exp_avg_sq = {
        'layer1': torch.tensor([[0.01, 0.02], [0.03, 0.04]]),
    }

    # 计算不同方法的得分
    scores_magnitude = compute_importance_scores_magnitude(weights)
    scores_first_order = compute_importance_scores_first_order(
        weights, gradients, exp_avg_sq
    )
    scores_adam = compute_importance_scores(
        weights, gradients, exp_avg_sq, alpha=0.5
    )

    # 验证形状一致
    assert scores_magnitude['layer1'].shape == scores_first_order['layer1'].shape
    assert scores_magnitude['layer1'].shape == scores_adam['layer1'].shape

    # 验证 magnitude 方法最简单（只依赖权重）
    assert torch.allclose(
        scores_magnitude['layer1'],
        torch.abs(weights['layer1'])
    )

    # 验证不同方法的结果不同
    assert not torch.allclose(scores_magnitude['layer1'], scores_first_order['layer1'])
    assert not torch.allclose(scores_magnitude['layer1'], scores_adam['layer1'])


def test_magnitude_scores_ranking():
    """测试基于 magnitude 的排序。"""
    weights = {
        'layer1': torch.tensor([1.0, -3.0, 2.0, -4.0, 0.5]),
    }

    scores = compute_importance_scores_magnitude(weights)
    flat_scores = scores['layer1']

    # 获取排序后的索引
    sorted_indices = torch.argsort(flat_scores, descending=True)

    # 验证排序（绝对值最大的应该排在前面）
    expected_order = [3, 1, 2, 0, 4]  # 对应 [-4.0, -3.0, 2.0, 1.0, 0.5]
    assert sorted_indices.tolist() == expected_order


def test_magnitude_scores_with_zeros():
    """测试包含零值的情况。"""
    weights = {
        'layer1': torch.tensor([1.0, 0.0, -2.0, 0.0, 3.0]),
    }

    scores = compute_importance_scores_magnitude(weights)

    # 零值的得分应该是 0
    assert scores['layer1'][1] == 0.0
    assert scores['layer1'][3] == 0.0

    # 非零值的得分应该是绝对值
    assert scores['layer1'][0] == 1.0
    assert scores['layer1'][2] == 2.0
    assert scores['layer1'][4] == 3.0


def test_magnitude_scores_large_model():
    """测试大模型的情况。"""
    # 模拟一个较大的模型
    weights = {
        f'layer{i}.weight': torch.randn(100, 100)
        for i in range(10)
    }

    scores = compute_importance_scores_magnitude(weights)

    # 验证所有层都有得分
    assert len(scores) == 10

    # 验证所有得分都是非负的
    for name, score in scores.items():
        assert torch.all(score >= 0)
        assert score.shape == weights[name].shape


def test_magnitude_scores_flattened():
    """测试展平后的得分。"""
    weights = {
        'layer1': torch.tensor([[1.0, -2.0], [3.0, -4.0]]),
        'layer2': torch.tensor([0.5, -1.5]),
    }

    scores = compute_importance_scores_magnitude(weights)
    flat_scores = get_flattened_scores(scores)

    # 验证展平后的形状
    expected_size = 2 * 2 + 2  # layer1: 4 elements, layer2: 2 elements
    assert flat_scores.shape == (expected_size,)

    # 验证所有值都是非负的
    assert torch.all(flat_scores >= 0)


def test_magnitude_method_documentation():
    """测试方法文档是否完整。"""
    import inspect

    # 获取函数文档
    doc = inspect.getdoc(compute_importance_scores_magnitude)

    # 验证文档包含关键信息
    assert doc is not None
    assert '公式' in doc or 'formula' in doc.lower()
    assert 'θ' in doc or 'weight' in doc.lower()
    assert '参数' in doc or 'parameters' in doc.lower()
    assert '返回' in doc or 'returns' in doc.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
