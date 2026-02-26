"""
Inshrinkerator 端到端压缩和解压测试。
"""

import pytest
import torch
from typing import Dict

from ckpt_compress.methods.inshrinkerator.inshrinkerator import InshrinkeratorCompressor


class TestInshrinkeratorRoundtrip:
    """Inshrinkerator 端到端压缩/解压测试。"""

    def test_inshrinkerator_roundtrip_single_checkpoint(self):
        """压缩和解压应该保持结构。"""
        torch.manual_seed(42)

        W_t = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
        }
        grad = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
        }

        compressor = InshrinkeratorCompressor()
        compressed = compressor.compress(W_t, grad, prev_quantized=None)
        W_hat = compressor.decompress(compressed, prev_quantized=None)

        assert set(W_hat.keys()) == set(W_t.keys())
        for key in W_t:
            assert W_hat[key].shape == W_t[key].shape

    def test_inshrinkerator_chain_two_checkpoints(self):
        """测试链式压缩: W_1 -> W_2。"""
        torch.manual_seed(42)

        W_1 = {"weight": torch.randn(32, 32)}
        grad_1 = {"weight": torch.randn(32, 32)}

        W_2 = {"weight": W_1["weight"] + torch.randn(32, 32) * 0.1}
        grad_2 = {"weight": torch.randn(32, 32)}

        compressor = InshrinkeratorCompressor()

        # 压缩第一个检查点
        compressed_1 = compressor.compress(W_1, grad_1, prev_quantized=None)
        W_1_hat = compressor.decompress(compressed_1, prev_quantized=None)

        # 获取用于增量编码的量化表示
        q_1 = compressor.get_quantized_indices(compressed_1)

        # 使用第一个作为参考压缩第二个检查点
        compressed_2 = compressor.compress(W_2, grad_2, prev_quantized=q_1)
        W_2_hat = compressor.decompress(compressed_2, prev_quantized=q_1)

        assert W_1_hat["weight"].shape == W_1["weight"].shape
        assert W_2_hat["weight"].shape == W_2["weight"].shape

    def test_inshrinkerator_compressed_smaller(self):
        """压缩数据应该比原始数据小。"""
        torch.manual_seed(42)

        W_t = {
            "layer1.weight": torch.randn(256, 128),
            "layer2.weight": torch.randn(128, 64),
        }
        grad = {
            "layer1.weight": torch.randn(256, 128),
            "layer2.weight": torch.randn(128, 64),
        }

        compressor = InshrinkeratorCompressor()
        compressed = compressor.compress(W_t, grad, prev_quantized=None)

        original_size = sum(t.numel() * 4 for t in W_t.values())
        compressed_size = len(compressed)

        assert compressed_size < original_size


class TestInshrinkeratorQuality:
    """Inshrinkerator 压缩质量测试。"""

    def test_inshrinkerator_reconstruction_error_bounded(self):
        """重建误差应该有界。"""
        torch.manual_seed(42)

        W_t = {"weight": torch.randn(64, 64)}
        grad = {"weight": torch.randn(64, 64)}

        compressor = InshrinkeratorCompressor()
        compressed = compressor.compress(W_t, grad, prev_quantized=None)
        W_hat = compressor.decompress(compressed, prev_quantized=None)

        mse = torch.mean((W_t["weight"] - W_hat["weight"]) ** 2)
        variance = torch.var(W_t["weight"])
        relative_error = mse / variance

        # 误差应该合理
        assert relative_error < 0.3

    def test_inshrinkerator_preserves_dtype(self):
        """输出张量应该有正确的 dtype。"""
        W_t = {"weight": torch.randn(32, 32, dtype=torch.float32)}
        grad = {"weight": torch.randn(32, 32, dtype=torch.float32)}

        compressor = InshrinkeratorCompressor()
        compressed = compressor.compress(W_t, grad, prev_quantized=None)
        W_hat = compressor.decompress(compressed, prev_quantized=None)

        assert W_hat["weight"].dtype == torch.float32
