"""
预测残差压缩 - 优化器状态压缩模块测试。

TDD：先写测试，再实现。

关键特性：
- 压缩 Adam 优化器状态（exp_avg, exp_avg_sq）
- 利用优化器状态的结构特性实现更好的压缩
- 支持检查点之间的增量编码
"""

import pytest
import torch

from ckpt_compress.methods.predictive.optimizer_compress import (
    compress_exp_avg,
    decompress_exp_avg,
    compress_exp_avg_sq,
    decompress_exp_avg_sq,
    OptimizerStateCompressor,
)


class TestCompressExpAvg:
    """一阶动量（exp_avg）压缩测试。"""

    def test_compress_exp_avg_basic(self):
        """应压缩 exp_avg 张量。"""
        exp_avg = torch.randn(32, 32) * 0.1

        compressed = compress_exp_avg(exp_avg, n_bits=8)

        assert "indices" in compressed
        assert "scale" in compressed
        assert "shape" in compressed

    def test_compress_decompress_exp_avg_roundtrip(self):
        """压缩-解压应近似原始值。"""
        torch.manual_seed(42)
        exp_avg = torch.randn(64, 64) * 0.1

        compressed = compress_exp_avg(exp_avg, n_bits=8)
        reconstructed = decompress_exp_avg(compressed)

        assert reconstructed.shape == exp_avg.shape
        mse = torch.mean((exp_avg - reconstructed) ** 2)
        assert mse < 0.01  # 合理误差

    def test_compress_exp_avg_preserves_sign(self):
        """应保持 exp_avg 值的符号。"""
        exp_avg = torch.tensor([-0.5, 0.3, -0.1, 0.8])

        compressed = compress_exp_avg(exp_avg, n_bits=8)
        reconstructed = decompress_exp_avg(compressed)

        # 符号应匹配
        assert torch.all(torch.sign(reconstructed) == torch.sign(exp_avg))


class TestCompressExpAvgSq:
    """二阶动量（exp_avg_sq）压缩测试。"""

    def test_compress_exp_avg_sq_basic(self):
        """应压缩 exp_avg_sq 张量。"""
        # exp_avg_sq 始终为非负
        exp_avg_sq = torch.abs(torch.randn(32, 32)) * 0.01

        compressed = compress_exp_avg_sq(exp_avg_sq, n_bits=8)

        assert "indices" in compressed
        assert "scale" in compressed
        assert "shape" in compressed

    def test_compress_decompress_exp_avg_sq_roundtrip(self):
        """压缩-解压应近似原始值。"""
        torch.manual_seed(42)
        exp_avg_sq = torch.abs(torch.randn(64, 64)) * 0.01

        compressed = compress_exp_avg_sq(exp_avg_sq, n_bits=8)
        reconstructed = decompress_exp_avg_sq(compressed)

        assert reconstructed.shape == exp_avg_sq.shape
        mse = torch.mean((exp_avg_sq - reconstructed) ** 2)
        assert mse < 0.001  # 应该准确

    def test_compress_exp_avg_sq_non_negative(self):
        """重建的 exp_avg_sq 应为非负。"""
        torch.manual_seed(42)
        exp_avg_sq = torch.abs(torch.randn(32, 32)) * 0.01

        compressed = compress_exp_avg_sq(exp_avg_sq, n_bits=8)
        reconstructed = decompress_exp_avg_sq(compressed)

        assert torch.all(reconstructed >= 0)

    def test_compress_exp_avg_sq_log_scale(self):
        """应使用对数尺度以更好地压缩小值。"""
        # exp_avg_sq 通常有非常小的值
        exp_avg_sq = torch.tensor([1e-8, 1e-6, 1e-4, 1e-2])

        compressed = compress_exp_avg_sq(exp_avg_sq, n_bits=8)
        reconstructed = decompress_exp_avg_sq(compressed)

        # 所有尺度的相对误差应合理
        relative_error = torch.abs(exp_avg_sq - reconstructed) / (exp_avg_sq + 1e-10)
        assert torch.all(relative_error < 0.5)  # 50% 相对误差以内


class TestOptimizerStateCompressor:
    """OptimizerStateCompressor 类测试。"""

    def test_compressor_init(self):
        """压缩器应使用配置初始化。"""
        compressor = OptimizerStateCompressor(n_bits=8)

        assert compressor.n_bits == 8

    def test_compressor_compress_single_param(self):
        """应压缩单个参数的优化器状态。"""
        torch.manual_seed(42)
        compressor = OptimizerStateCompressor(n_bits=8)

        optimizer_state = {
            "weight": {
                "exp_avg": torch.randn(32, 32) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(32, 32)) * 0.01,
            }
        }

        compressed = compressor.compress(optimizer_state)

        assert "weight" in compressed
        assert "exp_avg" in compressed["weight"]
        assert "exp_avg_sq" in compressed["weight"]

    def test_compressor_decompress_single_param(self):
        """应解压单个参数的优化器状态。"""
        torch.manual_seed(42)
        compressor = OptimizerStateCompressor(n_bits=8)

        optimizer_state = {
            "weight": {
                "exp_avg": torch.randn(32, 32) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(32, 32)) * 0.01,
            }
        }

        compressed = compressor.compress(optimizer_state)
        reconstructed = compressor.decompress(compressed)

        assert "weight" in reconstructed
        assert "exp_avg" in reconstructed["weight"]
        assert "exp_avg_sq" in reconstructed["weight"]
        assert reconstructed["weight"]["exp_avg"].shape == optimizer_state["weight"]["exp_avg"].shape
        assert reconstructed["weight"]["exp_avg_sq"].shape == optimizer_state["weight"]["exp_avg_sq"].shape

    def test_compressor_roundtrip_multiple_params(self):
        """应处理多个参数。"""
        torch.manual_seed(42)
        compressor = OptimizerStateCompressor(n_bits=8)

        optimizer_state = {
            "layer1.weight": {
                "exp_avg": torch.randn(64, 32) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(64, 32)) * 0.01,
            },
            "layer1.bias": {
                "exp_avg": torch.randn(64) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(64)) * 0.01,
            },
            "layer2.weight": {
                "exp_avg": torch.randn(32, 64) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(32, 64)) * 0.01,
            },
        }

        compressed = compressor.compress(optimizer_state)
        reconstructed = compressor.decompress(compressed)

        assert set(reconstructed.keys()) == set(optimizer_state.keys())
        for key in optimizer_state:
            assert reconstructed[key]["exp_avg"].shape == optimizer_state[key]["exp_avg"].shape
            assert reconstructed[key]["exp_avg_sq"].shape == optimizer_state[key]["exp_avg_sq"].shape

    def test_compressor_compression_ratio(self):
        """压缩后大小应小于原始大小。"""
        torch.manual_seed(42)
        compressor = OptimizerStateCompressor(n_bits=4)

        optimizer_state = {
            "weight": {
                "exp_avg": torch.randn(256, 256) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(256, 256)) * 0.01,
            }
        }

        compressed = compressor.compress(optimizer_state)

        # 原始大小：2 * 256 * 256 * 4 字节 = 512 KB
        original_size = 2 * 256 * 256 * 4

        # 压缩后大小（4 比特 = 每个值 0.5 字节）
        compressed_size = (
            compressed["weight"]["exp_avg"]["indices"].numel() +
            compressed["weight"]["exp_avg_sq"]["indices"].numel()
        ) * 0.5  # 4-bit

        assert compressed_size < original_size

    def test_compressor_delta_encoding(self):
        """应支持检查点之间的增量编码。"""
        torch.manual_seed(42)
        compressor = OptimizerStateCompressor(n_bits=8, use_delta=True)

        # 第一个检查点
        state_1 = {
            "weight": {
                "exp_avg": torch.randn(32, 32) * 0.1,
                "exp_avg_sq": torch.abs(torch.randn(32, 32)) * 0.01,
            }
        }

        # 第二个检查点（与第一个相似）
        state_2 = {
            "weight": {
                "exp_avg": state_1["weight"]["exp_avg"] + torch.randn(32, 32) * 0.01,
                "exp_avg_sq": state_1["weight"]["exp_avg_sq"] + torch.abs(torch.randn(32, 32)) * 0.001,
            }
        }

        # 压缩第一个检查点
        compressed_1 = compressor.compress(state_1, prev_state=None)

        # 使用增量压缩第二个检查点
        compressed_2 = compressor.compress(state_2, prev_state=state_1)

        # 解压
        reconstructed_1 = compressor.decompress(compressed_1, prev_state=None)
        reconstructed_2 = compressor.decompress(compressed_2, prev_state=reconstructed_1)

        # 应正确重建
        assert reconstructed_2["weight"]["exp_avg"].shape == state_2["weight"]["exp_avg"].shape

    def test_compressor_empty_state(self):
        """应处理空优化器状态。"""
        compressor = OptimizerStateCompressor(n_bits=8)

        optimizer_state = {}

        compressed = compressor.compress(optimizer_state)
        reconstructed = compressor.decompress(compressed)

        assert reconstructed == {}
