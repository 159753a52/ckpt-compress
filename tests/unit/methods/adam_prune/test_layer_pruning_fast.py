"""
分层剪枝快速实现测试 (TDD)。

测试向量化的分层剪枝和全局剪枝函数。
"""

import pytest
import torch
import numpy as np
from typing import Dict


# ============================================================
# 测试 prune_by_layer_sparsity (分层剪枝)
# ============================================================
class TestPruneByLayerSparsity:
    """测试分层剪枝函数。"""

    def test_basic_layer_pruning(self):
        """基本分层剪枝测试。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_layer_sparsity,
        )

        weights = {
            "layer1": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            "layer2": torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]),
        }
        importance_scores = {
            "layer1": torch.tensor([0.1, 0.5, 0.3, 0.2, 0.4]),
            "layer2": torch.tensor([0.5, 0.1, 0.4, 0.3, 0.2]),
        }
        # 每层剪枝 40% (2个参数)
        layer_ratios = {"layer1": 0.4, "layer2": 0.4}

        pruned_weights, masks = prune_by_layer_sparsity(
            weights, importance_scores, layer_ratios
        )

        # 检查每层都剪枝了 2 个参数
        assert (masks["layer1"] == 0).sum() == 2
        assert (masks["layer2"] == 0).sum() == 2

        # 检查剪枝的是重要性最低的参数
        # layer1: 最低的是 index 0 (0.1) 和 index 3 (0.2)
        assert masks["layer1"][0] == 0
        assert masks["layer1"][3] == 0

        # layer2: 最低的是 index 1 (0.1) 和 index 4 (0.2)
        assert masks["layer2"][1] == 0
        assert masks["layer2"][4] == 0

    def test_zero_sparsity(self):
        """稀疏度为0时不应剪枝。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_layer_sparsity,
        )

        weights = {"layer": torch.randn(100)}
        importance_scores = {"layer": torch.rand(100)}
        layer_ratios = {"layer": 0.0}

        pruned_weights, masks = prune_by_layer_sparsity(
            weights, importance_scores, layer_ratios
        )

        # 没有参数被剪枝
        assert (masks["layer"] == 0).sum() == 0
        torch.testing.assert_close(pruned_weights["layer"], weights["layer"])

    def test_full_sparsity(self):
        """稀疏度为1时应剪枝所有参数。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_layer_sparsity,
        )

        weights = {"layer": torch.randn(100)}
        importance_scores = {"layer": torch.rand(100)}
        layer_ratios = {"layer": 1.0}

        pruned_weights, masks = prune_by_layer_sparsity(
            weights, importance_scores, layer_ratios
        )

        # 所有参数被剪枝
        assert (masks["layer"] == 0).sum() == 100
        assert (pruned_weights["layer"] == 0).all()

    def test_different_ratios_per_layer(self):
        """不同层可以有不同的剪枝比例。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_layer_sparsity,
        )

        weights = {
            "layer1": torch.randn(100),
            "layer2": torch.randn(200),
            "layer3": torch.randn(50),
        }
        importance_scores = {
            "layer1": torch.rand(100),
            "layer2": torch.rand(200),
            "layer3": torch.rand(50),
        }
        layer_ratios = {
            "layer1": 0.1,  # 剪枝 10 个
            "layer2": 0.5,  # 剪枝 100 个
            "layer3": 0.2,  # 剪枝 10 个
        }

        pruned_weights, masks = prune_by_layer_sparsity(
            weights, importance_scores, layer_ratios
        )

        assert (masks["layer1"] == 0).sum() == 10
        assert (masks["layer2"] == 0).sum() == 100
        assert (masks["layer3"] == 0).sum() == 10

    def test_missing_layer_ratio_defaults_to_zero(self):
        """缺失的层剪枝比例默认为0。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_layer_sparsity,
        )

        weights = {
            "layer1": torch.randn(100),
            "layer2": torch.randn(100),
        }
        importance_scores = {
            "layer1": torch.rand(100),
            "layer2": torch.rand(100),
        }
        # 只指定 layer1 的剪枝比例
        layer_ratios = {"layer1": 0.5}

        pruned_weights, masks = prune_by_layer_sparsity(
            weights, importance_scores, layer_ratios
        )

        assert (masks["layer1"] == 0).sum() == 50
        assert (masks["layer2"] == 0).sum() == 0  # 默认不剪枝

    def test_preserves_tensor_shape(self):
        """应保持张量形状不变。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_layer_sparsity,
        )

        weights = {
            "conv": torch.randn(64, 32, 3, 3),
            "linear": torch.randn(256, 128),
        }
        importance_scores = {
            "conv": torch.rand(64, 32, 3, 3),
            "linear": torch.rand(256, 128),
        }
        layer_ratios = {"conv": 0.3, "linear": 0.5}

        pruned_weights, masks = prune_by_layer_sparsity(
            weights, importance_scores, layer_ratios
        )

        assert pruned_weights["conv"].shape == (64, 32, 3, 3)
        assert pruned_weights["linear"].shape == (256, 128)
        assert masks["conv"].shape == (64, 32, 3, 3)
        assert masks["linear"].shape == (256, 128)

    def test_gpu_tensor_support(self):
        """应支持 GPU tensor。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_layer_sparsity,
        )

        weights = {"layer": torch.randn(100)}
        importance_scores = {"layer": torch.rand(100)}

        if torch.cuda.is_available():
            weights = {k: v.cuda() for k, v in weights.items()}
            importance_scores = {k: v.cuda() for k, v in importance_scores.items()}

        layer_ratios = {"layer": 0.3}

        pruned_weights, masks = prune_by_layer_sparsity(
            weights, importance_scores, layer_ratios
        )

        assert (masks["layer"] == 0).sum() == 30

    def test_empty_dict(self):
        """空字典应返回空结果。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_layer_sparsity,
        )

        pruned_weights, masks = prune_by_layer_sparsity({}, {}, {})

        assert pruned_weights == {}
        assert masks == {}


# ============================================================
# 测试 prune_by_global_sparsity_fast (快速全局剪枝)
# ============================================================
class TestPruneByGlobalSparsityFast:
    """测试快速全局剪枝函数。"""

    def test_basic_global_pruning(self):
        """基本全局剪枝测试。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_global_sparsity_fast,
        )

        weights = {
            "layer1": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            "layer2": torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]),
        }
        importance_scores = {
            "layer1": torch.tensor([0.1, 0.5, 0.3, 0.2, 0.4]),
            "layer2": torch.tensor([0.5, 0.1, 0.4, 0.3, 0.2]),
        }

        # 全局剪枝 40% (4个参数)
        pruned_weights, masks = prune_by_global_sparsity_fast(
            weights, importance_scores, 0.4
        )

        # 检查总共剪枝了 4 个参数
        total_pruned = sum((m == 0).sum().item() for m in masks.values())
        assert total_pruned == 4

    def test_zero_sparsity(self):
        """稀疏度为0时不应剪枝。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_global_sparsity_fast,
        )

        weights = {"layer": torch.randn(100)}
        importance_scores = {"layer": torch.rand(100)}

        pruned_weights, masks = prune_by_global_sparsity_fast(
            weights, importance_scores, 0.0
        )

        assert (masks["layer"] == 0).sum() == 0

    def test_full_sparsity(self):
        """稀疏度为1时应剪枝所有参数。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_global_sparsity_fast,
        )

        weights = {"layer": torch.randn(100)}
        importance_scores = {"layer": torch.rand(100)}

        pruned_weights, masks = prune_by_global_sparsity_fast(
            weights, importance_scores, 1.0
        )

        assert (masks["layer"] == 0).sum() == 100

    def test_invalid_sparsity_raises_error(self):
        """无效稀疏度应抛出错误。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_global_sparsity_fast,
        )

        weights = {"layer": torch.randn(100)}
        importance_scores = {"layer": torch.rand(100)}

        with pytest.raises(ValueError):
            prune_by_global_sparsity_fast(weights, importance_scores, -0.1)

        with pytest.raises(ValueError):
            prune_by_global_sparsity_fast(weights, importance_scores, 1.1)

    def test_preserves_tensor_shape(self):
        """应保持张量形状不变。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_global_sparsity_fast,
        )

        weights = {
            "conv": torch.randn(64, 32, 3, 3),
            "linear": torch.randn(256, 128),
        }
        importance_scores = {
            "conv": torch.rand(64, 32, 3, 3),
            "linear": torch.rand(256, 128),
        }

        pruned_weights, masks = prune_by_global_sparsity_fast(
            weights, importance_scores, 0.3
        )

        assert pruned_weights["conv"].shape == (64, 32, 3, 3)
        assert pruned_weights["linear"].shape == (256, 128)

    def test_performance_large_model(self):
        """大模型应能快速处理。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_global_sparsity_fast,
        )
        import time

        # 模拟大模型 (约 10M 参数)
        weights = {
            f"layer{i}": torch.randn(1000, 1000) for i in range(10)
        }
        importance_scores = {
            f"layer{i}": torch.rand(1000, 1000) for i in range(10)
        }

        start = time.time()
        pruned_weights, masks = prune_by_global_sparsity_fast(
            weights, importance_scores, 0.5
        )
        elapsed = time.time() - start

        # 应该在 5 秒内完成 (向量化实现)
        assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s"

    def test_gpu_tensor_support(self):
        """应支持 GPU tensor。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_global_sparsity_fast,
        )

        weights = {"layer": torch.randn(100)}
        importance_scores = {"layer": torch.rand(100)}

        if torch.cuda.is_available():
            weights = {k: v.cuda() for k, v in weights.items()}
            importance_scores = {k: v.cuda() for k, v in importance_scores.items()}

        pruned_weights, masks = prune_by_global_sparsity_fast(
            weights, importance_scores, 0.3
        )

        assert (masks["layer"] == 0).sum() == 30

    def test_matches_original_implementation(self):
        """结果应与原始实现一致。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_global_sparsity,
            prune_by_global_sparsity_fast,
        )

        torch.manual_seed(42)
        weights = {
            "layer1": torch.randn(50),
            "layer2": torch.randn(50),
        }
        importance_scores = {
            "layer1": torch.rand(50),
            "layer2": torch.rand(50),
        }

        # 原始实现
        pruned_orig, masks_orig = prune_by_global_sparsity(
            weights, importance_scores, 0.3
        )

        # 快速实现
        pruned_fast, masks_fast = prune_by_global_sparsity_fast(
            weights, importance_scores, 0.3
        )

        # 剪枝数量应该相同
        orig_pruned = sum((m == 0).sum().item() for m in masks_orig.values())
        fast_pruned = sum((m == 0).sum().item() for m in masks_fast.values())
        assert orig_pruned == fast_pruned


# ============================================================
# 测试 compute_layer_sparsities_from_global (从全局稀疏度计算分层稀疏度)
# ============================================================
class TestComputeLayerSparsitiesFromGlobal:
    """测试从全局稀疏度计算分层稀疏度。"""

    def test_uniform_importance(self):
        """均匀重要性时各层稀疏度应相近。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            compute_layer_sparsities_from_global,
        )

        # 所有层重要性相同
        layer_stats = {
            "layer1": {"n_params": 100, "mean_importance": 1.0},
            "layer2": {"n_params": 100, "mean_importance": 1.0},
            "layer3": {"n_params": 100, "mean_importance": 1.0},
        }

        ratios = compute_layer_sparsities_from_global(layer_stats, 0.3)

        # 各层稀疏度应该接近 0.3
        for name, ratio in ratios.items():
            assert abs(ratio - 0.3) < 0.1

    def test_varying_importance(self):
        """不同重要性时低重要性层应剪枝更多。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            compute_layer_sparsities_from_global,
        )

        layer_stats = {
            "low_importance": {"n_params": 100, "mean_importance": 0.1},
            "high_importance": {"n_params": 100, "mean_importance": 1.0},
        }

        ratios = compute_layer_sparsities_from_global(layer_stats, 0.5)

        # 低重要性层应该剪枝更多
        assert ratios["low_importance"] > ratios["high_importance"]

    def test_zero_global_sparsity(self):
        """全局稀疏度为0时各层稀疏度也应为0。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            compute_layer_sparsities_from_global,
        )

        layer_stats = {
            "layer1": {"n_params": 100, "mean_importance": 1.0},
            "layer2": {"n_params": 100, "mean_importance": 0.5},
        }

        ratios = compute_layer_sparsities_from_global(layer_stats, 0.0)

        assert ratios["layer1"] == 0.0
        assert ratios["layer2"] == 0.0

    def test_respects_global_sparsity_constraint(self):
        """总剪枝参数数应接近全局稀疏度。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            compute_layer_sparsities_from_global,
        )

        layer_stats = {
            "layer1": {"n_params": 1000, "mean_importance": 0.5},
            "layer2": {"n_params": 2000, "mean_importance": 1.0},
            "layer3": {"n_params": 500, "mean_importance": 0.2},
        }
        target_sparsity = 0.4

        ratios = compute_layer_sparsities_from_global(layer_stats, target_sparsity)

        # 计算实际全局稀疏度
        total_params = sum(s["n_params"] for s in layer_stats.values())
        total_pruned = sum(
            layer_stats[name]["n_params"] * ratio
            for name, ratio in ratios.items()
        )
        actual_sparsity = total_pruned / total_params

        # 应该接近目标稀疏度 (允许 10% 误差)
        assert abs(actual_sparsity - target_sparsity) < 0.1


# ============================================================
# 测试 prune_by_layer_distribution (基于分布的分层剪枝)
# ============================================================
class TestPruneByLayerDistribution:
    """测试基于分布参数的分层剪枝。"""

    def test_basic_distribution_pruning(self):
        """基本分布剪枝测试。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_layer_distribution,
        )

        weights = {
            "layer1": torch.randn(100),
            "layer2": torch.randn(200),
        }
        importance_scores = {
            "layer1": torch.rand(100),
            "layer2": torch.rand(200),
        }
        layer_dist_params = {
            "layer1": {"N": 100, "dist_type": "weibull", "k": 1.5, "beta": 0.1, "loc": 0.0},
            "layer2": {"N": 200, "dist_type": "weibull", "k": 1.2, "beta": 0.2, "loc": 0.0},
        }

        pruned_weights, masks, ratios = prune_by_layer_distribution(
            weights, importance_scores, layer_dist_params, target_sparsity=0.3
        )

        # 检查返回了剪枝比例
        assert "layer1" in ratios
        assert "layer2" in ratios

        # 检查全局稀疏度接近目标
        total_params = 300
        total_pruned = sum((m == 0).sum().item() for m in masks.values())
        actual_sparsity = total_pruned / total_params
        assert abs(actual_sparsity - 0.3) < 0.15  # 允许 15% 误差

    def test_returns_layer_ratios(self):
        """应返回每层的剪枝比例。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_layer_distribution,
        )

        weights = {"layer": torch.randn(100)}
        importance_scores = {"layer": torch.rand(100)}
        layer_dist_params = {
            "layer": {"N": 100, "dist_type": "weibull", "k": 1.0, "beta": 0.1, "loc": 0.0},
        }

        pruned_weights, masks, ratios = prune_by_layer_distribution(
            weights, importance_scores, layer_dist_params, target_sparsity=0.5
        )

        assert "layer" in ratios
        assert 0 <= ratios["layer"] <= 1


# ============================================================
# 测试边界情况
# ============================================================
class TestEdgeCases:
    """测试边界情况。"""

    def test_single_element_tensor(self):
        """单元素张量应能处理。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_layer_sparsity,
        )

        weights = {"layer": torch.tensor([1.0])}
        importance_scores = {"layer": torch.tensor([0.5])}
        layer_ratios = {"layer": 1.0}

        pruned_weights, masks = prune_by_layer_sparsity(
            weights, importance_scores, layer_ratios
        )

        assert masks["layer"].item() == 0

    def test_very_small_sparsity(self):
        """非常小的稀疏度应能处理。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_layer_sparsity,
        )

        weights = {"layer": torch.randn(1000)}
        importance_scores = {"layer": torch.rand(1000)}
        layer_ratios = {"layer": 0.001}  # 剪枝 1 个参数

        pruned_weights, masks = prune_by_layer_sparsity(
            weights, importance_scores, layer_ratios
        )

        assert (masks["layer"] == 0).sum() == 1

    def test_negative_importance_scores(self):
        """负重要性分数应能处理。"""
        from ckpt_compress.methods.adam_prune.layer_pruning import (
            prune_by_layer_sparsity,
        )

        weights = {"layer": torch.randn(100)}
        # 包含负值的重要性分数
        importance_scores = {"layer": torch.randn(100)}
        layer_ratios = {"layer": 0.3}

        pruned_weights, masks = prune_by_layer_sparsity(
            weights, importance_scores, layer_ratios
        )

        # 应该剪枝重要性最低的（包括负值）
        assert (masks["layer"] == 0).sum() == 30
