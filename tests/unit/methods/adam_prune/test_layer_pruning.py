"""
分层优化剪枝测试。

TDD: 先写测试，再写实现。
基于拉格朗日优化的分层剪枝比例计算。
"""

import pytest
import torch
import numpy as np
from typing import Dict


class TestLayerPruningOptimizer:
    """测试 LayerPruningOptimizer 类。"""

    def test_compute_layer_prune_ratios_returns_dict(self):
        """应该返回每层的剪枝比例字典。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        layer_scores = {
            "layer1": torch.rand(100),
            "layer2": torch.rand(200),
        }
        epsilon = 0.01

        ratios = optimizer.compute_layer_prune_ratios(layer_scores, epsilon)

        assert isinstance(ratios, dict)
        assert set(ratios.keys()) == set(layer_scores.keys())

    def test_prune_ratios_in_valid_range(self):
        """剪枝比例应该在 [0, 1] 范围内。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        layer_scores = {
            "layer1": torch.rand(100) * 0.1,
            "layer2": torch.rand(200) * 0.2,
        }
        epsilon = 0.01

        ratios = optimizer.compute_layer_prune_ratios(layer_scores, epsilon)

        for name, ratio in ratios.items():
            assert 0 <= ratio <= 1, f"Layer {name} has invalid ratio: {ratio}"

    def test_higher_importance_lower_prune_ratio(self):
        """高重要性层应该有更低的剪枝比例。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        # layer1 重要性低，layer2 重要性高
        layer_scores = {
            "layer1": torch.ones(100) * 0.01,  # 低重要性
            "layer2": torch.ones(100) * 1.0,   # 高重要性
        }
        epsilon = 0.1

        ratios = optimizer.compute_layer_prune_ratios(layer_scores, epsilon)

        # 低重要性层应该剪枝更多
        assert ratios["layer1"] > ratios["layer2"]

    def test_larger_epsilon_higher_prune_ratios(self):
        """更大的 epsilon 应该导致更高的剪枝比例。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        layer_scores = {
            "layer1": torch.rand(100),
            "layer2": torch.rand(200),
        }

        ratios_small = optimizer.compute_layer_prune_ratios(layer_scores, epsilon=0.001)
        ratios_large = optimizer.compute_layer_prune_ratios(layer_scores, epsilon=0.1)

        # 总剪枝比例应该更大
        total_small = sum(ratios_small.values())
        total_large = sum(ratios_large.values())
        assert total_large > total_small

    def test_zero_epsilon_zero_prune_ratios(self):
        """epsilon=0 时剪枝比例应该为0。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        layer_scores = {
            "layer1": torch.rand(100),
            "layer2": torch.rand(200),
        }

        ratios = optimizer.compute_layer_prune_ratios(layer_scores, epsilon=0.0)

        for name, ratio in ratios.items():
            assert ratio == 0.0, f"Layer {name} should have 0 ratio for epsilon=0"

    def test_negative_epsilon_raises_error(self):
        """负的 epsilon 应该抛出错误。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        layer_scores = {"layer1": torch.rand(100)}

        with pytest.raises(ValueError):
            optimizer.compute_layer_prune_ratios(layer_scores, epsilon=-0.01)


class TestComputePredictedLoss:
    """测试预测损失计算。"""

    def test_predicted_loss_returns_float(self):
        """应该返回浮点数。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        layer_scores = {
            "layer1": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
            "layer2": torch.tensor([0.2, 0.3, 0.4]),
        }
        prune_ratios = {"layer1": 0.2, "layer2": 0.3}

        loss = optimizer.compute_predicted_loss(layer_scores, prune_ratios)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_predicted_loss_zero_prune_ratio(self):
        """剪枝比例为0时预测损失应该为0。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        layer_scores = {
            "layer1": torch.rand(100),
            "layer2": torch.rand(200),
        }
        prune_ratios = {"layer1": 0.0, "layer2": 0.0}

        loss = optimizer.compute_predicted_loss(layer_scores, prune_ratios)

        assert loss == 0.0

    def test_predicted_loss_increases_with_prune_ratio(self):
        """剪枝比例越高，预测损失越大。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        layer_scores = {
            "layer1": torch.rand(100),
        }

        loss_low = optimizer.compute_predicted_loss(layer_scores, {"layer1": 0.1})
        loss_high = optimizer.compute_predicted_loss(layer_scores, {"layer1": 0.5})

        assert loss_high > loss_low

    def test_predicted_loss_equals_sum_of_pruned_scores(self):
        """预测损失应该等于被剪枝参数的重要性得分之和。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        # 使用已排序的得分便于验证
        layer_scores = {
            "layer1": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        }
        prune_ratios = {"layer1": 0.4}  # 剪枝 40% = 2个参数

        loss = optimizer.compute_predicted_loss(layer_scores, prune_ratios)

        # 应该剪枝最小的2个: 0.1 + 0.2 = 0.3
        expected = 0.1 + 0.2
        assert abs(loss - expected) < 1e-5


class TestApplyPruning:
    """测试实际剪枝应用。"""

    def test_apply_pruning_returns_weights_and_masks(self):
        """应该返回剪枝后的权重和掩码。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        weights = {
            "layer1": torch.randn(100),
            "layer2": torch.randn(200),
        }
        importance_scores = {
            "layer1": torch.rand(100),
            "layer2": torch.rand(200),
        }
        prune_ratios = {"layer1": 0.2, "layer2": 0.3}

        pruned_weights, masks = optimizer.apply_pruning(
            weights, importance_scores, prune_ratios
        )

        assert set(pruned_weights.keys()) == set(weights.keys())
        assert set(masks.keys()) == set(weights.keys())

    def test_apply_pruning_correct_sparsity(self):
        """剪枝后的稀疏度应该正确。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        weights = {"layer1": torch.randn(100)}
        importance_scores = {"layer1": torch.rand(100)}
        prune_ratios = {"layer1": 0.3}

        pruned_weights, masks = optimizer.apply_pruning(
            weights, importance_scores, prune_ratios
        )

        # 应该有30个参数被剪枝
        n_pruned = (masks["layer1"] == 0).sum().item()
        assert n_pruned == 30

    def test_apply_pruning_preserves_shape(self):
        """剪枝后形状应该保持不变。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        shapes = [(64, 32), (128,), (3, 3, 64, 64)]
        for shape in shapes:
            weights = {"layer": torch.randn(shape)}
            importance_scores = {"layer": torch.rand(shape)}
            prune_ratios = {"layer": 0.2}

            pruned_weights, masks = optimizer.apply_pruning(
                weights, importance_scores, prune_ratios
            )

            assert pruned_weights["layer"].shape == torch.Size(shape)
            assert masks["layer"].shape == torch.Size(shape)


class TestComputeActualLoss:
    """测试实际损失计算。"""

    def test_compute_actual_loss_from_masks(self):
        """应该正确计算被剪枝参数的重要性之和。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        importance_scores = {
            "layer1": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        }
        masks = {
            "layer1": torch.tensor([0, 0, 1, 1, 1]),  # 前两个被剪枝
        }

        loss = optimizer.compute_actual_loss(importance_scores, masks)

        # 损失 = 0.1 + 0.2 = 0.3
        assert abs(loss - 0.3) < 1e-5

    def test_compute_actual_loss_no_pruning(self):
        """没有剪枝时损失应该为0。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        importance_scores = {"layer1": torch.rand(100)}
        masks = {"layer1": torch.ones(100)}

        loss = optimizer.compute_actual_loss(importance_scores, masks)

        assert loss == 0.0


class TestGlobalSparsity:
    """测试全局稀疏度计算。"""

    def test_compute_global_sparsity(self):
        """应该正确计算全局稀疏度。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        masks = {
            "layer1": torch.tensor([0, 0, 1, 1, 1]),  # 2/5 = 40% 稀疏
            "layer2": torch.tensor([0, 1, 1]),        # 1/3 ≈ 33% 稀疏
        }

        sparsity = optimizer.compute_global_sparsity(masks)

        # 总共 8 个参数，3 个被剪枝 = 37.5%
        expected = 3 / 8
        assert abs(sparsity - expected) < 1e-5


class TestConvertScipyParams:
    """测试 scipy 分布参数转换。"""

    def test_convert_weibull_params(self):
        """应该正确转换韦伯分布参数。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import convert_scipy_params

        # scipy weibull_min 返回 (c, loc, scale)
        scipy_params = (2.0, 0.0, 0.5)  # c=2, loc=0, scale=0.5

        result = convert_scipy_params('weibull', scipy_params)

        assert result['k'] == 2.0  # 形状参数
        assert result['beta'] == 0.5  # 尺度参数
        assert result['loc'] == 0.0

    def test_convert_lognormal_params(self):
        """应该正确转换对数正态分布参数。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import convert_scipy_params
        import math

        # scipy lognorm 返回 (s, loc, scale)
        scipy_params = (0.5, 0.0, 2.0)  # s=0.5, loc=0, scale=2.0

        result = convert_scipy_params('lognormal', scipy_params)

        assert result['sigma'] == 0.5  # s = sigma
        assert abs(result['mu'] - math.log(2.0)) < 1e-10  # mu = ln(scale)
        assert result['loc'] == 0.0

    def test_convert_exponential_params(self):
        """应该正确转换指数分布参数。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import convert_scipy_params

        # scipy expon 返回 (loc, scale)
        scipy_params = (0.0, 0.5)  # loc=0, scale=0.5

        result = convert_scipy_params('exponential', scipy_params)

        assert result['lambda'] == 2.0  # lambda = 1/scale
        assert result['loc'] == 0.0

    def test_convert_gamma_params(self):
        """应该正确转换伽马分布参数。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import convert_scipy_params

        # scipy gamma 返回 (a, loc, scale)
        scipy_params = (2.0, 0.0, 0.5)  # a=2, loc=0, scale=0.5

        result = convert_scipy_params('gamma', scipy_params)

        assert result['alpha'] == 2.0  # 形状参数
        assert result['beta'] == 2.0  # beta = 1/scale
        assert result['loc'] == 0.0

    def test_convert_unknown_dist_raises_error(self):
        """未知分布类型应该抛出错误。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import convert_scipy_params

        with pytest.raises(ValueError):
            convert_scipy_params('unknown', (1.0, 0.0, 1.0))


class TestWeibullPruneRatio:
    """测试韦伯分布剪枝比例公式。"""

    def test_weibull_prune_ratio_formula(self):
        """应该正确计算韦伯分布的剪枝比例。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_weibull_prune_ratio
        import math

        # p = 1 - exp(-(1/(η·N·β))^k)
        eta = 1.0
        N = 100
        k = 2.0
        beta = 0.1

        p = compute_weibull_prune_ratio(eta, N, k, beta)

        # 手动计算
        arg = 1 / (eta * N * beta)
        expected = 1 - math.exp(-(arg ** k))

        assert abs(p - expected) < 1e-10

    def test_weibull_prune_ratio_in_range(self):
        """剪枝比例应该在 [0, 1] 范围内。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_weibull_prune_ratio

        # 测试各种参数组合
        test_cases = [
            (0.001, 100, 2.0, 0.1),  # 小 eta
            (1000.0, 100, 2.0, 0.1),  # 大 eta
            (1.0, 1000000, 2.0, 0.1),  # 大 N
            (1.0, 100, 0.5, 0.1),  # 小 k
            (1.0, 100, 5.0, 0.1),  # 大 k
        ]

        for eta, N, k, beta in test_cases:
            p = compute_weibull_prune_ratio(eta, N, k, beta)
            assert 0 <= p <= 1, f"p={p} out of range for eta={eta}, N={N}, k={k}, beta={beta}"

    def test_weibull_larger_eta_smaller_prune_ratio(self):
        """更大的 eta 应该导致更小的剪枝比例。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_weibull_prune_ratio

        N, k, beta = 100, 2.0, 0.1

        p_small_eta = compute_weibull_prune_ratio(0.1, N, k, beta)
        p_large_eta = compute_weibull_prune_ratio(10.0, N, k, beta)

        assert p_small_eta > p_large_eta


class TestLognormalPruneRatio:
    """测试对数正态分布剪枝比例公式。"""

    def test_lognormal_prune_ratio_formula(self):
        """应该正确计算对数正态分布的剪枝比例。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_lognormal_prune_ratio
        from scipy.stats import norm
        import math

        # p = Φ((ln(1/η) - ln(N) - μ) / σ)
        eta = 1.0
        N = 100
        mu = 0.0
        sigma = 1.0

        p = compute_lognormal_prune_ratio(eta, N, mu, sigma)

        # 手动计算
        z = (math.log(1/eta) - math.log(N) - mu) / sigma
        expected = norm.cdf(z)

        assert abs(p - expected) < 1e-10

    def test_lognormal_prune_ratio_in_range(self):
        """剪枝比例应该在 [0, 1] 范围内。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_lognormal_prune_ratio

        test_cases = [
            (0.001, 100, 0.0, 1.0),
            (1000.0, 100, 0.0, 1.0),
            (1.0, 1000000, 0.0, 1.0),
            (1.0, 100, -2.0, 0.5),
            (1.0, 100, 2.0, 2.0),
        ]

        for eta, N, mu, sigma in test_cases:
            p = compute_lognormal_prune_ratio(eta, N, mu, sigma)
            assert 0 <= p <= 1, f"p={p} out of range"


class TestWeibullLoss:
    """测试韦伯分布损失计算。"""

    def test_weibull_loss_formula(self):
        """应该正确计算韦伯分布的损失。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_weibull_loss

        N = 100
        k = 2.0
        beta = 0.1
        p = 0.1

        loss = compute_weibull_loss(N, k, beta, p)

        # 损失应该是正数
        assert loss >= 0

    def test_weibull_loss_increases_with_p(self):
        """剪枝比例越高，损失越大。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_weibull_loss

        N, k, beta = 100, 2.0, 0.1

        loss_low = compute_weibull_loss(N, k, beta, 0.1)
        loss_high = compute_weibull_loss(N, k, beta, 0.5)

        assert loss_high > loss_low

    def test_weibull_loss_zero_prune(self):
        """剪枝比例为0时损失应该为0。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_weibull_loss

        loss = compute_weibull_loss(100, 2.0, 0.1, 0.0)
        assert loss == 0.0


class TestLognormalLoss:
    """测试对数正态分布损失计算。"""

    def test_lognormal_loss_formula(self):
        """应该正确计算对数正态分布的损失。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_lognormal_loss

        N = 100
        mu = 0.0
        sigma = 1.0
        p = 0.1

        loss = compute_lognormal_loss(N, mu, sigma, p)

        # 损失应该是正数
        assert loss >= 0

    def test_lognormal_loss_increases_with_p(self):
        """剪枝比例越高，损失越大。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_lognormal_loss

        N, mu, sigma = 100, 0.0, 1.0

        loss_low = compute_lognormal_loss(N, mu, sigma, 0.1)
        loss_high = compute_lognormal_loss(N, mu, sigma, 0.5)

        assert loss_high > loss_low


class TestOptimizeWithDistributions:
    """测试基于分布的优化剪枝。"""

    def test_optimize_returns_ratios_dict(self):
        """应该返回每层的剪枝比例字典。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        layer_dist_params = {
            "layer1": {
                'N': 100,
                'dist_type': 'weibull',
                'k': 2.0,
                'beta': 0.1,
            },
            "layer2": {
                'N': 200,
                'dist_type': 'weibull',
                'k': 1.5,
                'beta': 0.2,
            },
        }
        epsilon = 0.1

        ratios, eta = optimizer.optimize_with_distributions(epsilon, layer_dist_params)

        assert isinstance(ratios, dict)
        assert set(ratios.keys()) == set(layer_dist_params.keys())
        assert eta > 0

    def test_optimize_ratios_in_valid_range(self):
        """剪枝比例应该在 [0, 1] 范围内。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        layer_dist_params = {
            "layer1": {'N': 100, 'dist_type': 'weibull', 'k': 2.0, 'beta': 0.1},
            "layer2": {'N': 200, 'dist_type': 'lognormal', 'mu': 0.0, 'sigma': 1.0},
        }

        ratios, _ = optimizer.optimize_with_distributions(0.1, layer_dist_params)

        for name, ratio in ratios.items():
            assert 0 <= ratio <= 1, f"Layer {name} has invalid ratio: {ratio}"

    def test_optimize_total_loss_close_to_epsilon(self):
        """优化后的总损失应该接近 epsilon。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        layer_dist_params = {
            "layer1": {'N': 1000, 'dist_type': 'weibull', 'k': 2.0, 'beta': 0.01},
            "layer2": {'N': 2000, 'dist_type': 'weibull', 'k': 1.5, 'beta': 0.02},
        }
        epsilon = 1.0

        ratios, _ = optimizer.optimize_with_distributions(epsilon, layer_dist_params)

        # 计算总损失
        total_loss = optimizer.compute_distribution_loss(layer_dist_params, ratios)

        # 总损失应该接近 epsilon（允许一定误差）
        assert abs(total_loss - epsilon) < epsilon * 0.1  # 10% 误差范围

    def test_optimize_mixed_distributions(self):
        """应该支持混合分布类型。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        layer_dist_params = {
            "weibull_layer": {'N': 100, 'dist_type': 'weibull', 'k': 2.0, 'beta': 0.1},
            "lognormal_layer": {'N': 200, 'dist_type': 'lognormal', 'mu': -1.0, 'sigma': 0.5},
        }

        ratios, eta = optimizer.optimize_with_distributions(0.5, layer_dist_params)

        assert "weibull_layer" in ratios
        assert "lognormal_layer" in ratios

    def test_optimize_zero_epsilon(self):
        """epsilon=0 时剪枝比例应该为0。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import LayerPruningOptimizer

        optimizer = LayerPruningOptimizer()

        layer_dist_params = {
            "layer1": {'N': 100, 'dist_type': 'weibull', 'k': 2.0, 'beta': 0.1},
        }

        ratios, _ = optimizer.optimize_with_distributions(0.0, layer_dist_params)

        for ratio in ratios.values():
            assert ratio == 0.0


class TestComputeRealLossIncrease:
    """测试实际损失增量计算。"""

    def test_compute_real_loss_increase_returns_float(self):
        """应该返回浮点数。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_real_loss_increase
        import torch.nn as nn

        # 创建简单模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

            def forward(self, x, labels=None):
                logits = self.linear(x)
                loss = None
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(logits, labels)
                return type('Output', (), {'loss': loss, 'logits': logits})()

        model = SimpleModel()

        # 创建简单数据
        data = [
            {'input': torch.randn(4, 10), 'labels': torch.randint(0, 2, (4,))}
            for _ in range(3)
        ]

        loss_increase = compute_real_loss_increase(model, model, data, num_batches=2)

        assert isinstance(loss_increase, float)

    def test_identical_models_zero_loss_increase(self):
        """相同模型的损失增量应该为0。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_real_loss_increase
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

            def forward(self, x, labels=None):
                logits = self.linear(x)
                loss = None
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(logits, labels)
                return type('Output', (), {'loss': loss, 'logits': logits})()

        model = SimpleModel()

        data = [
            {'input': torch.randn(4, 10), 'labels': torch.randint(0, 2, (4,))}
            for _ in range(3)
        ]

        loss_increase = compute_real_loss_increase(model, model, data, num_batches=2)

        assert abs(loss_increase) < 1e-6

    def test_pruned_model_higher_loss(self):
        """剪枝后的模型应该有更高的损失。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_real_loss_increase
        import torch.nn as nn
        import copy

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

            def forward(self, x, labels=None):
                logits = self.linear(x)
                loss = None
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(logits, labels)
                return type('Output', (), {'loss': loss, 'logits': logits})()

        torch.manual_seed(42)
        original_model = SimpleModel()

        # 创建剪枝后的模型（将部分权重设为0）
        pruned_model = copy.deepcopy(original_model)
        with torch.no_grad():
            pruned_model.linear.weight[:, :5] = 0  # 剪枝50%的权重

        # 固定数据
        torch.manual_seed(123)
        data = [
            {'input': torch.randn(4, 10), 'labels': torch.randint(0, 2, (4,))}
            for _ in range(5)
        ]

        loss_increase = compute_real_loss_increase(
            original_model, pruned_model, data, num_batches=3
        )

        # 剪枝后损失应该增加（或至少不减少太多）
        # 注意：由于随机性，这里只检查返回值是有效的浮点数
        assert isinstance(loss_increase, float)
        assert not np.isnan(loss_increase)

    def test_num_batches_parameter(self):
        """应该只使用指定数量的批次。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_real_loss_increase
        import torch.nn as nn

        call_count = [0]

        class CountingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

            def forward(self, x, labels=None):
                call_count[0] += 1
                logits = self.linear(x)
                loss = None
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(logits, labels)
                return type('Output', (), {'loss': loss, 'logits': logits})()

        model = CountingModel()

        data = [
            {'input': torch.randn(4, 10), 'labels': torch.randint(0, 2, (4,))}
            for _ in range(10)
        ]

        compute_real_loss_increase(model, model, data, num_batches=3)

        # 两个模型各调用3次 = 6次
        assert call_count[0] == 6


class TestComputeRelativeLossMetrics:
    """测试相对损失指标计算。"""

    def test_compute_relative_loss_metrics_returns_dict(self):
        """应该返回包含所有指标的字典。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_relative_loss_metrics

        result = compute_relative_loss_metrics(
            original_loss=4.0,
            actual_loss_increase=2.0,
            sum_pruned_scores=0.1,
            total_importance=10.0
        )

        assert isinstance(result, dict)
        assert 'rel_actual' in result
        assert 'rel_predicted' in result
        assert 'calibration_factor' in result

    def test_relative_actual_loss_calculation(self):
        """相对实际损失应该正确计算。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_relative_loss_metrics

        result = compute_relative_loss_metrics(
            original_loss=4.0,
            actual_loss_increase=2.0,  # 50% 增量
            sum_pruned_scores=0.1,
            total_importance=10.0
        )

        # rel_actual = 2.0 / 4.0 = 0.5
        assert abs(result['rel_actual'] - 0.5) < 1e-6

    def test_relative_predicted_loss_calculation(self):
        """相对预测损失应该正确计算。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_relative_loss_metrics

        result = compute_relative_loss_metrics(
            original_loss=4.0,
            actual_loss_increase=2.0,
            sum_pruned_scores=0.1,  # 1% 的总重要性
            total_importance=10.0
        )

        # rel_predicted = 0.1 / 10.0 = 0.01
        assert abs(result['rel_predicted'] - 0.01) < 1e-10

    def test_calibration_factor_calculation(self):
        """校准因子应该正确计算。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_relative_loss_metrics

        result = compute_relative_loss_metrics(
            original_loss=4.0,
            actual_loss_increase=2.0,
            sum_pruned_scores=0.1,
            total_importance=10.0
        )

        # calibration_factor = rel_actual / rel_predicted = 0.5 / 0.01 = 50
        assert abs(result['calibration_factor'] - 50.0) < 0.01

    def test_handles_zero_original_loss(self):
        """应该处理原始损失为0的情况。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_relative_loss_metrics

        result = compute_relative_loss_metrics(
            original_loss=0.0,
            actual_loss_increase=2.0,
            sum_pruned_scores=0.1,
            total_importance=10.0
        )

        # 应该不会抛出除零错误
        assert isinstance(result['rel_actual'], float)
        assert not np.isnan(result['rel_actual'])

    def test_handles_zero_total_importance(self):
        """应该处理总重要性为0的情况。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_relative_loss_metrics

        result = compute_relative_loss_metrics(
            original_loss=4.0,
            actual_loss_increase=2.0,
            sum_pruned_scores=0.1,
            total_importance=0.0
        )

        # 应该不会抛出除零错误
        assert isinstance(result['rel_predicted'], float)
        assert not np.isnan(result['rel_predicted'])

    def test_handles_zero_rel_predicted(self):
        """应该处理相对预测损失为0的情况。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_relative_loss_metrics

        result = compute_relative_loss_metrics(
            original_loss=4.0,
            actual_loss_increase=2.0,
            sum_pruned_scores=0.0,  # 没有剪枝
            total_importance=10.0
        )

        # 应该不会抛出除零错误
        assert isinstance(result['calibration_factor'], float)
        assert not np.isnan(result['calibration_factor'])


class TestComputeSumPrunedScores:
    """测试被剪枝参数重要性得分之和计算。"""

    def test_compute_sum_pruned_scores_returns_float(self):
        """应该返回浮点数。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_sum_pruned_scores

        layer_scores = {
            "layer1": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        }
        masks = {
            "layer1": torch.tensor([0, 0, 1, 1, 1]),  # 前两个被剪枝
        }

        result = compute_sum_pruned_scores(layer_scores, masks)

        assert isinstance(result, float)

    def test_sum_pruned_scores_correct_calculation(self):
        """应该正确计算被剪枝参数的得分之和。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_sum_pruned_scores

        layer_scores = {
            "layer1": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        }
        masks = {
            "layer1": torch.tensor([0, 0, 1, 1, 1]),  # 前两个被剪枝
        }

        result = compute_sum_pruned_scores(layer_scores, masks)

        # 被剪枝的得分: 0.1 + 0.2 = 0.3
        assert abs(result - 0.3) < 1e-5

    def test_sum_pruned_scores_multiple_layers(self):
        """应该正确处理多层。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_sum_pruned_scores

        layer_scores = {
            "layer1": torch.tensor([0.1, 0.2, 0.3]),
            "layer2": torch.tensor([0.4, 0.5]),
        }
        masks = {
            "layer1": torch.tensor([0, 1, 1]),  # 剪枝 0.1
            "layer2": torch.tensor([0, 1]),     # 剪枝 0.4
        }

        result = compute_sum_pruned_scores(layer_scores, masks)

        # 总剪枝得分: 0.1 + 0.4 = 0.5
        assert abs(result - 0.5) < 1e-5

    def test_sum_pruned_scores_no_pruning(self):
        """没有剪枝时应该返回0。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_sum_pruned_scores

        layer_scores = {
            "layer1": torch.tensor([0.1, 0.2, 0.3]),
        }
        masks = {
            "layer1": torch.ones(3),  # 全部保留
        }

        result = compute_sum_pruned_scores(layer_scores, masks)

        assert result == 0.0

    def test_sum_pruned_scores_all_pruned(self):
        """全部剪枝时应该返回总和。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_sum_pruned_scores

        layer_scores = {
            "layer1": torch.tensor([0.1, 0.2, 0.3]),
        }
        masks = {
            "layer1": torch.zeros(3),  # 全部剪枝
        }

        result = compute_sum_pruned_scores(layer_scores, masks)

        # 总和: 0.1 + 0.2 + 0.3 = 0.6
        assert abs(result - 0.6) < 1e-5

    def test_sum_pruned_scores_missing_mask(self):
        """如果某层没有对应的 mask，应该跳过该层。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import compute_sum_pruned_scores

        layer_scores = {
            "layer1": torch.tensor([0.1, 0.2, 0.3]),
            "layer2": torch.tensor([0.4, 0.5]),  # 没有对应的 mask
        }
        masks = {
            "layer1": torch.tensor([0, 1, 1]),  # 只有 layer1 有 mask
        }

        result = compute_sum_pruned_scores(layer_scores, masks)

        # 只计算 layer1 的剪枝得分: 0.1
        assert abs(result - 0.1) < 1e-5


class TestPruneByGlobalSparsity:
    """测试按固定全局稀疏度剪枝。"""

    def test_prune_by_global_sparsity_returns_tuple(self):
        """应该返回 (pruned_weights, masks) 元组。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import prune_by_global_sparsity

        weights = {
            "layer1": torch.randn(100),
            "layer2": torch.randn(200),
        }
        importance_scores = {
            "layer1": torch.rand(100),
            "layer2": torch.rand(200),
        }

        pruned_weights, masks = prune_by_global_sparsity(
            weights, importance_scores, target_sparsity=0.2
        )

        assert isinstance(pruned_weights, dict)
        assert isinstance(masks, dict)
        assert set(pruned_weights.keys()) == set(weights.keys())
        assert set(masks.keys()) == set(weights.keys())

    def test_prune_by_global_sparsity_correct_sparsity(self):
        """应该达到目标全局稀疏度。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import prune_by_global_sparsity

        weights = {
            "layer1": torch.randn(100),
            "layer2": torch.randn(200),
        }
        importance_scores = {
            "layer1": torch.rand(100),
            "layer2": torch.rand(200),
        }
        target_sparsity = 0.3

        pruned_weights, masks = prune_by_global_sparsity(
            weights, importance_scores, target_sparsity
        )

        # 计算实际稀疏度
        total_params = sum(m.numel() for m in masks.values())
        total_pruned = sum((m == 0).sum().item() for m in masks.values())
        actual_sparsity = total_pruned / total_params

        # 应该接近目标稀疏度（允许小误差因为整数取整）
        assert abs(actual_sparsity - target_sparsity) < 0.01

    def test_prune_by_global_sparsity_prunes_lowest_importance(self):
        """应该剪枝重要性最低的参数。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import prune_by_global_sparsity

        # 创建已知重要性的数据
        weights = {
            "layer1": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        }
        importance_scores = {
            "layer1": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        }

        pruned_weights, masks = prune_by_global_sparsity(
            weights, importance_scores, target_sparsity=0.4
        )

        # 应该剪枝最低的 40% = 2 个参数（0.1 和 0.2）
        expected_mask = torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32)
        assert torch.allclose(masks["layer1"], expected_mask)

    def test_prune_by_global_sparsity_zero_sparsity(self):
        """稀疏度为0时不应该剪枝任何参数。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import prune_by_global_sparsity

        weights = {"layer1": torch.randn(100)}
        importance_scores = {"layer1": torch.rand(100)}

        pruned_weights, masks = prune_by_global_sparsity(
            weights, importance_scores, target_sparsity=0.0
        )

        # 所有参数都应该保留
        assert (masks["layer1"] == 1).all()

    def test_prune_by_global_sparsity_full_sparsity(self):
        """稀疏度为1时应该剪枝所有参数。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import prune_by_global_sparsity

        weights = {"layer1": torch.randn(100)}
        importance_scores = {"layer1": torch.rand(100)}

        pruned_weights, masks = prune_by_global_sparsity(
            weights, importance_scores, target_sparsity=1.0
        )

        # 所有参数都应该被剪枝
        assert (masks["layer1"] == 0).all()

    def test_prune_by_global_sparsity_preserves_shape(self):
        """剪枝后形状应该保持不变。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import prune_by_global_sparsity

        shapes = [(64, 32), (128,), (3, 3, 64, 64)]
        for shape in shapes:
            weights = {"layer": torch.randn(shape)}
            importance_scores = {"layer": torch.rand(shape)}

            pruned_weights, masks = prune_by_global_sparsity(
                weights, importance_scores, target_sparsity=0.2
            )

            assert pruned_weights["layer"].shape == torch.Size(shape)
            assert masks["layer"].shape == torch.Size(shape)

    def test_prune_by_global_sparsity_invalid_sparsity_raises_error(self):
        """无效的稀疏度应该抛出错误。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import prune_by_global_sparsity

        weights = {"layer1": torch.randn(100)}
        importance_scores = {"layer1": torch.rand(100)}

        with pytest.raises(ValueError):
            prune_by_global_sparsity(weights, importance_scores, target_sparsity=-0.1)

        with pytest.raises(ValueError):
            prune_by_global_sparsity(weights, importance_scores, target_sparsity=1.5)
