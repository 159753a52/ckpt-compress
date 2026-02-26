"""
ExCP 压缩方法测试。
"""

import pytest
import torch
from typing import Dict

from ckpt_compress.methods.excp.residual import compute_residual, reconstruct


class TestResidual:
    """ExCP 残差计算测试。"""

    def test_residual_zero_when_equal(self):
        """当 W_t 等于 W_prev_hat 时，残差应为零。"""
        W_prev_hat = torch.tensor([1.0, 2.0, 3.0])
        W_t = torch.tensor([1.0, 2.0, 3.0])

        dW = compute_residual(W_prev_hat, W_t)

        assert torch.allclose(dW, torch.zeros_like(dW))

    def test_residual_sign(self):
        """残差对于正负变化应有正确的符号。"""
        W_prev_hat = torch.tensor([1.0, 2.0, 3.0])
        W_t = torch.tensor([2.0, 1.0, 3.0])  # +1, -1, 0

        dW = compute_residual(W_prev_hat, W_t)

        expected = torch.tensor([1.0, -1.0, 0.0])
        assert torch.allclose(dW, expected)

    def test_reconstruct_additivity(self):
        """重建应满足 W_hat = W_prev_hat + dW。"""
        W_prev_hat = torch.tensor([1.0, 2.0, 3.0])
        W_t = torch.tensor([1.5, 2.5, 2.5])

        dW = compute_residual(W_prev_hat, W_t)
        W_hat = reconstruct(W_prev_hat, dW)

        # 量化前应完全相等
        assert torch.allclose(W_hat, W_t)

    def test_residual_uses_reconstructed_prev(self):
        """残差应使用重建（hat）版本，而非原始版本。"""
        # 此测试验证 API 期望 W_prev_hat，而非 W_prev_raw
        W_prev_hat = torch.tensor([1.0, 2.0, 3.0])  # 这是重建版本
        W_t = torch.tensor([1.1, 2.1, 3.1])

        dW = compute_residual(W_prev_hat, W_t)

        # dW = W_t - W_prev_hat
        expected = W_t - W_prev_hat
        assert torch.allclose(dW, expected)

    def test_residual_dtype_consistency(self):
        """输出 dtype 应与输入 dtype 匹配。"""
        W_prev_hat = torch.tensor([1.0, 2.0], dtype=torch.float32)
        W_t = torch.tensor([1.5, 2.5], dtype=torch.float32)

        dW = compute_residual(W_prev_hat, W_t)

        assert dW.dtype == torch.float32

    def test_residual_2d_tensor(self):
        """应支持二维张量（权重矩阵）。"""
        W_prev_hat = torch.randn(64, 32)
        W_t = W_prev_hat + torch.randn(64, 32) * 0.1

        dW = compute_residual(W_prev_hat, W_t)
        W_hat = reconstruct(W_prev_hat, dW)

        assert dW.shape == W_prev_hat.shape
        assert torch.allclose(W_hat, W_t)

    def test_residual_preserves_shape(self):
        """残差应保持张量形状。"""
        shapes = [(10,), (10, 20), (2, 3, 4), (2, 3, 4, 5)]

        for shape in shapes:
            W_prev_hat = torch.randn(shape)
            W_t = torch.randn(shape)

            dW = compute_residual(W_prev_hat, W_t)

            assert dW.shape == torch.Size(shape)


class TestResidualStateDict:
    """状态字典残差计算测试。"""

    def test_compute_residual_state_dict(self):
        """应计算整个状态字典的残差。"""
        from ckpt_compress.methods.excp.residual import compute_residual_state_dict

        prev_hat = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
        }
        current = {
            "layer1.weight": prev_hat["layer1.weight"] + torch.randn(64, 32) * 0.1,
            "layer1.bias": prev_hat["layer1.bias"] + torch.randn(64) * 0.1,
        }

        residual = compute_residual_state_dict(prev_hat, current)

        assert set(residual.keys()) == set(prev_hat.keys())
        for key in residual:
            expected = current[key] - prev_hat[key]
            assert torch.allclose(residual[key], expected)

    def test_reconstruct_state_dict(self):
        """应从残差重建状态字典。"""
        from ckpt_compress.methods.excp.residual import (
            compute_residual_state_dict,
            reconstruct_state_dict,
        )

        prev_hat = {
            "weight": torch.randn(32, 32),
            "bias": torch.randn(32),
        }
        current = {
            "weight": prev_hat["weight"] + torch.randn(32, 32) * 0.1,
            "bias": prev_hat["bias"] + torch.randn(32) * 0.1,
        }

        residual = compute_residual_state_dict(prev_hat, current)
        reconstructed = reconstruct_state_dict(prev_hat, residual)

        for key in current:
            assert torch.allclose(reconstructed[key], current[key])
