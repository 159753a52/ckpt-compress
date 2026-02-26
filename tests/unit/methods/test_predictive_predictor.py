"""
预测残差压缩 - 预测器模块测试。

TDD：先写测试，再实现。
"""

import pytest
import torch

from ckpt_compress.methods.predictive.predictor import (
    sgd_predict,
    adam_predict,
    compute_prediction_residual,
    AdamPredictor,
)


class TestSGDPredict:
    """SGD 权重预测测试。"""

    def test_sgd_predict_basic(self):
        """SGD 预测：W_pred = W - lr * grad。"""
        W = torch.tensor([1.0, 2.0, 3.0])
        grad = torch.tensor([0.1, 0.2, 0.3])
        lr = 0.1

        W_pred = sgd_predict(W, grad, lr)

        # W_pred = W - lr * grad = [1-0.01, 2-0.02, 3-0.03]
        expected = torch.tensor([0.99, 1.98, 2.97])
        assert torch.allclose(W_pred, expected, atol=1e-6)

    def test_sgd_predict_zero_grad(self):
        """零梯度时，预测等于原始值。"""
        W = torch.tensor([1.0, 2.0, 3.0])
        grad = torch.zeros(3)
        lr = 0.1

        W_pred = sgd_predict(W, grad, lr)

        assert torch.allclose(W_pred, W)

    def test_sgd_predict_2d_tensor(self):
        """应支持二维张量。"""
        W = torch.randn(4, 4)
        grad = torch.randn(4, 4)
        lr = 0.01

        W_pred = sgd_predict(W, grad, lr)

        expected = W - lr * grad
        assert torch.allclose(W_pred, expected)


class TestAdamPredict:
    """Adam 权重预测测试。"""

    def test_adam_predict_basic(self):
        """Adam 预测：W_pred = W - lr * m / (sqrt(v) + eps)。"""
        W = torch.tensor([1.0, 2.0, 3.0])
        m = torch.tensor([0.1, 0.2, 0.3])  # exp_avg（一阶动量）
        v = torch.tensor([0.01, 0.04, 0.09])  # exp_avg_sq（二阶动量）
        lr = 0.001
        eps = 1e-8

        W_pred = adam_predict(W, m, v, lr, eps)

        # W_pred = W - lr * m / (sqrt(v) + eps)
        expected = W - lr * m / (torch.sqrt(v) + eps)
        assert torch.allclose(W_pred, expected, atol=1e-6)

    def test_adam_predict_zero_momentum(self):
        """零动量时，预测等于原始值。"""
        W = torch.tensor([1.0, 2.0, 3.0])
        m = torch.zeros(3)
        v = torch.tensor([0.01, 0.04, 0.09])
        lr = 0.001

        W_pred = adam_predict(W, m, v, lr)

        assert torch.allclose(W_pred, W, atol=1e-6)

    def test_adam_predict_2d_tensor(self):
        """应支持二维张量。"""
        torch.manual_seed(42)
        W = torch.randn(4, 4)
        m = torch.randn(4, 4) * 0.1
        v = torch.abs(torch.randn(4, 4)) * 0.01 + 1e-8  # v 必须为正
        lr = 0.001

        W_pred = adam_predict(W, m, v, lr)

        expected = W - lr * m / (torch.sqrt(v) + 1e-8)
        assert torch.allclose(W_pred, expected, atol=1e-6)


class TestComputePredictionResidual:
    """预测残差计算测试。"""

    def test_residual_basic(self):
        """残差 = W_actual - W_pred。"""
        W_actual = torch.tensor([1.0, 2.0, 3.0])
        W_pred = torch.tensor([0.9, 1.9, 2.9])

        residual = compute_prediction_residual(W_actual, W_pred)

        expected = torch.tensor([0.1, 0.1, 0.1])
        assert torch.allclose(residual, expected, atol=1e-6)

    def test_residual_zero_when_perfect_prediction(self):
        """完美预测时残差应为零。"""
        W = torch.tensor([1.0, 2.0, 3.0])

        residual = compute_prediction_residual(W, W)

        assert torch.allclose(residual, torch.zeros(3))

    def test_residual_smaller_than_delta(self):
        """预测残差应小于直接增量。"""
        torch.manual_seed(42)
        # 模拟 Adam 更新
        W_prev = torch.randn(100)
        m = torch.randn(100) * 0.1
        v = torch.abs(torch.randn(100)) * 0.01 + 1e-8
        lr = 0.001

        # 实际下一步权重（模拟 Adam 步骤）
        W_actual = W_prev - lr * m / (torch.sqrt(v) + 1e-8)
        # 添加小噪声模拟真实训练
        W_actual = W_actual + torch.randn(100) * 0.001

        # 预测
        W_pred = adam_predict(W_prev, m, v, lr)

        # 残差 vs 直接增量
        residual = compute_prediction_residual(W_actual, W_pred)
        direct_delta = W_actual - W_prev

        # 残差应更小（更利于压缩）
        assert residual.abs().mean() < direct_delta.abs().mean()


class TestAdamPredictor:
    """AdamPredictor 类测试。"""

    def test_predictor_init(self):
        """预测器应使用默认配置初始化。"""
        predictor = AdamPredictor()

        assert predictor.lr == 0.001
        assert predictor.eps == 1e-8
        assert predictor.beta1 == 0.9
        assert predictor.beta2 == 0.999

    def test_predictor_custom_config(self):
        """预测器应接受自定义配置。"""
        predictor = AdamPredictor(lr=0.01, eps=1e-6, beta1=0.8, beta2=0.99)

        assert predictor.lr == 0.01
        assert predictor.eps == 1e-6
        assert predictor.beta1 == 0.8
        assert predictor.beta2 == 0.99

    def test_predictor_predict_weights(self):
        """预测器应从状态字典预测权重。"""
        torch.manual_seed(42)
        predictor = AdamPredictor(lr=0.001)

        W_prev = {
            "layer1.weight": torch.randn(4, 4),
            "layer1.bias": torch.randn(4),
        }
        optimizer_state = {
            "layer1.weight": {
                "exp_avg": torch.randn(4, 4) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(4, 4)) * 0.01 + 1e-8,
            },
            "layer1.bias": {
                "exp_avg": torch.randn(4) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(4)) * 0.01 + 1e-8,
            },
        }

        W_pred = predictor.predict(W_prev, optimizer_state)

        assert set(W_pred.keys()) == set(W_prev.keys())
        for key in W_prev:
            assert W_pred[key].shape == W_prev[key].shape

    def test_predictor_compute_residual(self):
        """预测器应从实际值和预测值计算残差。"""
        torch.manual_seed(42)
        predictor = AdamPredictor(lr=0.001)

        W_prev = {"weight": torch.randn(4, 4)}
        optimizer_state = {
            "weight": {
                "exp_avg": torch.randn(4, 4) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(4, 4)) * 0.01 + 1e-8,
            }
        }
        W_actual = {"weight": torch.randn(4, 4)}

        residual = predictor.compute_residual(W_actual, W_prev, optimizer_state)

        assert "weight" in residual
        assert residual["weight"].shape == W_actual["weight"].shape

    def test_predictor_reconstruct_from_residual(self):
        """应从残差重建实际权重。"""
        torch.manual_seed(42)
        predictor = AdamPredictor(lr=0.001)

        W_prev = {"weight": torch.randn(4, 4)}
        optimizer_state = {
            "weight": {
                "exp_avg": torch.randn(4, 4) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(4, 4)) * 0.01 + 1e-8,
            }
        }
        W_actual = {"weight": torch.randn(4, 4)}

        # 计算残差
        residual = predictor.compute_residual(W_actual, W_prev, optimizer_state)

        # 重建
        W_reconstructed = predictor.reconstruct(residual, W_prev, optimizer_state)

        assert torch.allclose(W_reconstructed["weight"], W_actual["weight"], atol=1e-6)

    def test_predictor_fallback_to_delta_without_optimizer_state(self):
        """无优化器状态时，应回退到直接增量。"""
        torch.manual_seed(42)
        predictor = AdamPredictor()

        W_prev = {"weight": torch.randn(4, 4)}
        W_actual = {"weight": torch.randn(4, 4)}

        # 此键无优化器状态
        optimizer_state = {}

        residual = predictor.compute_residual(W_actual, W_prev, optimizer_state)

        # 应为直接增量
        expected = W_actual["weight"] - W_prev["weight"]
        assert torch.allclose(residual["weight"], expected)
