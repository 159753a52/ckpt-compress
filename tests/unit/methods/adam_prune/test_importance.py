"""
AdamPrune 重要性得分计算测试。

测试公式: s_i = -g_i * θ_i + α * v_i * θ_i²
"""

import pytest
import torch
from typing import Dict


class TestComputeImportanceScores:
    """测试 compute_importance_scores 函数。"""

    def test_importance_score_formula_correct(self):
        """验证重要性得分公式: s_i = -g_i * θ_i + α * v_i * θ_i²"""
        from ckpt_compress.methods.adam_prune.importance import compute_importance_scores

        # 创建简单的测试数据
        weights = {"layer.weight": torch.tensor([1.0, 2.0, -3.0])}
        gradients = {"layer.weight": torch.tensor([0.1, -0.2, 0.3])}
        exp_avg_sq = {"layer.weight": torch.tensor([0.01, 0.04, 0.09])}

        scores = compute_importance_scores(weights, gradients, exp_avg_sq, alpha=0.5)

        # 手动计算期望值 (保留符号)
        # s_0 = -0.1 * 1.0 + 0.5 * 0.01 * 1.0² = -0.1 + 0.005 = -0.095
        # s_1 = -(-0.2) * 2.0 + 0.5 * 0.04 * 2.0² = 0.4 + 0.08 = 0.48
        # s_2 = -0.3 * (-3.0) + 0.5 * 0.09 * (-3.0)² = 0.9 + 0.405 = 1.305
        expected = torch.tensor([-0.095, 0.48, 1.305])

        assert "layer.weight" in scores
        torch.testing.assert_close(scores["layer.weight"], expected, rtol=1e-5, atol=1e-5)

    def test_importance_score_all_zeros_weights(self):
        """全零权重应该返回全零得分。"""
        from ckpt_compress.methods.adam_prune.importance import compute_importance_scores

        weights = {"layer.weight": torch.zeros(10)}
        gradients = {"layer.weight": torch.randn(10)}
        exp_avg_sq = {"layer.weight": torch.rand(10)}

        scores = compute_importance_scores(weights, gradients, exp_avg_sq)

        assert torch.allclose(scores["layer.weight"], torch.zeros(10))

    def test_importance_score_all_zeros_gradients(self):
        """全零梯度时，得分应该只有二阶项。"""
        from ckpt_compress.methods.adam_prune.importance import compute_importance_scores

        weights = {"layer.weight": torch.tensor([1.0, 2.0, 3.0])}
        gradients = {"layer.weight": torch.zeros(3)}
        exp_avg_sq = {"layer.weight": torch.tensor([0.1, 0.2, 0.3])}

        scores = compute_importance_scores(weights, gradients, exp_avg_sq)

        # s_i = 0 + 0.5 * v_i * θ_i²
        expected = 0.5 * exp_avg_sq["layer.weight"] * weights["layer.weight"] ** 2
        torch.testing.assert_close(scores["layer.weight"], expected)

    def test_importance_score_shape_preserved(self):
        """输出形状应该与输入相同。"""
        from ckpt_compress.methods.adam_prune.importance import compute_importance_scores

        shapes = [(64, 32), (128,), (3, 3, 64, 64)]
        for shape in shapes:
            weights = {"layer": torch.randn(shape)}
            gradients = {"layer": torch.randn(shape)}
            exp_avg_sq = {"layer": torch.rand(shape)}

            scores = compute_importance_scores(weights, gradients, exp_avg_sq)

            assert scores["layer"].shape == torch.Size(shape)

    def test_importance_score_can_be_negative(self):
        """重要性得分可以为负（当一阶项为负且大于二阶项时）。"""
        from ckpt_compress.methods.adam_prune.importance import compute_importance_scores

        # 构造一个一阶项为负的情况: -g*θ < 0 当 g 和 θ 同号
        weights = {"layer": torch.tensor([1.0])}
        gradients = {"layer": torch.tensor([1.0])}  # g 和 θ 同号
        exp_avg_sq = {"layer": torch.tensor([0.01])}  # 小的二阶项

        scores = compute_importance_scores(weights, gradients, exp_avg_sq, alpha=0.5)

        # s = -1.0 * 1.0 + 0.5 * 0.01 * 1.0² = -1.0 + 0.005 = -0.995
        assert scores["layer"].item() < 0

    def test_importance_score_multiple_layers(self):
        """应该正确处理多个层。"""
        from ckpt_compress.methods.adam_prune.importance import compute_importance_scores

        weights = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
            "layer2.weight": torch.randn(128, 64),
        }
        gradients = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
            "layer2.weight": torch.randn(128, 64),
        }
        exp_avg_sq = {
            "layer1.weight": torch.rand(64, 32),
            "layer1.bias": torch.rand(64),
            "layer2.weight": torch.rand(128, 64),
        }

        scores = compute_importance_scores(weights, gradients, exp_avg_sq)

        assert set(scores.keys()) == set(weights.keys())
        for key in weights:
            assert scores[key].shape == weights[key].shape

    def test_importance_score_missing_key_raises_error(self):
        """如果键不匹配应该抛出错误。"""
        from ckpt_compress.methods.adam_prune.importance import compute_importance_scores

        weights = {"layer.weight": torch.randn(10)}
        gradients = {"layer.weight": torch.randn(10)}
        exp_avg_sq = {"other.weight": torch.rand(10)}  # 键不匹配

        with pytest.raises(KeyError):
            compute_importance_scores(weights, gradients, exp_avg_sq)

    def test_importance_score_different_dtypes(self):
        """应该处理不同的数据类型。"""
        from ckpt_compress.methods.adam_prune.importance import compute_importance_scores

        for dtype in [torch.float32, torch.float64]:
            weights = {"layer": torch.randn(10, dtype=dtype)}
            gradients = {"layer": torch.randn(10, dtype=dtype)}
            exp_avg_sq = {"layer": torch.rand(10, dtype=dtype)}

            scores = compute_importance_scores(weights, gradients, exp_avg_sq)

            assert scores["layer"].dtype == dtype

    def test_alpha_parameter_effect(self):
        """α参数应该影响二阶项的权重。"""
        from ckpt_compress.methods.adam_prune.importance import compute_importance_scores

        weights = {"layer": torch.tensor([2.0])}
        gradients = {"layer": torch.tensor([0.0])}  # 零梯度，只有二阶项
        exp_avg_sq = {"layer": torch.tensor([1.0])}

        scores_low = compute_importance_scores(weights, gradients, exp_avg_sq, alpha=0.1)
        scores_high = compute_importance_scores(weights, gradients, exp_avg_sq, alpha=1.0)

        # α越大，得分越高
        assert scores_high["layer"].item() > scores_low["layer"].item()
        # 比例应该是 1.0/0.1 = 10
        ratio = scores_high["layer"].item() / scores_low["layer"].item()
        assert abs(ratio - 10.0) < 1e-5

    def test_default_alpha(self):
        """默认 alpha 应该是 0.5。"""
        from ckpt_compress.methods.adam_prune.importance import compute_importance_scores

        weights = {"layer": torch.tensor([2.0])}
        gradients = {"layer": torch.tensor([0.0])}
        exp_avg_sq = {"layer": torch.tensor([1.0])}

        scores_default = compute_importance_scores(weights, gradients, exp_avg_sq)
        scores_explicit = compute_importance_scores(weights, gradients, exp_avg_sq, alpha=0.5)

        torch.testing.assert_close(scores_default["layer"], scores_explicit["layer"])


class TestGetFlattenedScores:
    """测试 get_flattened_scores 函数。"""

    def test_flatten_single_layer(self):
        """单层应该正确展平。"""
        from ckpt_compress.methods.adam_prune.importance import get_flattened_scores

        scores = {"layer": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        flat = get_flattened_scores(scores)

        assert flat.shape == (4,)
        torch.testing.assert_close(flat, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_flatten_multiple_layers(self):
        """多层应该正确拼接。"""
        from ckpt_compress.methods.adam_prune.importance import get_flattened_scores

        scores = {
            "layer1": torch.tensor([1.0, 2.0]),
            "layer2": torch.tensor([3.0, 4.0, 5.0]),
        }
        flat = get_flattened_scores(scores)

        assert flat.shape == (5,)

    def test_flatten_empty_dict(self):
        """空字典应该返回空张量。"""
        from ckpt_compress.methods.adam_prune.importance import get_flattened_scores

        scores = {}
        flat = get_flattened_scores(scores)

        assert flat.numel() == 0


class TestComputeMeanImportance:
    """测试 compute_mean_importance 函数。"""

    def test_mean_importance_correct(self):
        """平均重要性应该正确计算。"""
        from ckpt_compress.methods.adam_prune.importance import compute_mean_importance

        scores = {
            "layer1": torch.tensor([1.0, 2.0, 3.0]),
            "layer2": torch.tensor([4.0, 5.0]),
        }
        mean = compute_mean_importance(scores)

        # (1+2+3+4+5) / 5 = 3.0
        assert abs(mean - 3.0) < 1e-6

    def test_mean_importance_single_value(self):
        """单个值的平均应该是它本身。"""
        from ckpt_compress.methods.adam_prune.importance import compute_mean_importance

        scores = {"layer": torch.tensor([5.0])}
        mean = compute_mean_importance(scores)

        assert abs(mean - 5.0) < 1e-6

    def test_mean_importance_with_negative_values(self):
        """应该正确处理负值。"""
        from ckpt_compress.methods.adam_prune.importance import compute_mean_importance

        scores = {
            "layer": torch.tensor([-1.0, 0.0, 1.0]),
        }
        mean = compute_mean_importance(scores)

        # (-1+0+1) / 3 = 0.0
        assert abs(mean - 0.0) < 1e-6
