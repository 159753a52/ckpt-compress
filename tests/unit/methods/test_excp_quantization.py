"""
ExCP K-means 量化模块测试。
"""

import pytest
import torch

from ckpt_compress.methods.excp.quantization import (
    kmeans_quantize_nonzero,
    dequantize,
    pack_int4,
    unpack_int4,
)


class TestKmeansQuantize:
    """K-means 量化测试。"""

    def test_quantize_zero_stays_zero(self):
        """零值在量化后应保持为零。"""
        x = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0])
        n_bits = 4

        indices, centers = kmeans_quantize_nonzero(x, n_bits)
        x_hat = dequantize(indices, centers)

        # 零位置应该精确为零
        zero_mask = (x == 0)
        assert torch.all(x_hat[zero_mask] == 0)

    def test_centers_count_is_2pow_n_minus_1(self):
        """非零值应聚类为 2^n - 1 个中心。"""
        x = torch.randn(1000)  # 足够多的唯一值
        x[x.abs() < 0.1] = 0  # 添加一些零
        n_bits = 4

        indices, centers = kmeans_quantize_nonzero(x, n_bits)

        # 应该最多有 2^4 - 1 = 15 个非零中心（加上 0）
        # centers[0] 应该是 0
        assert len(centers) <= 2 ** n_bits
        assert centers[0] == 0  # 第一个中心始终为 0

    def test_indices_range(self):
        """索引应在 [0, 2^n - 1] 范围内。"""
        x = torch.randn(100)
        n_bits = 4

        indices, centers = kmeans_quantize_nonzero(x, n_bits)

        assert indices.min() >= 0
        assert indices.max() <= 2 ** n_bits - 1

    def test_dequantize_shape(self):
        """反量化张量应与输入形状相同。"""
        x = torch.randn(64, 32)
        n_bits = 4

        indices, centers = kmeans_quantize_nonzero(x, n_bits)
        x_hat = dequantize(indices, centers)

        assert x_hat.shape == x.shape

    def test_dequantize_zero_exact(self):
        """索引为零的元素应反量化为精确的零。"""
        x = torch.tensor([0.0, 0.0, 1.0, 2.0])
        n_bits = 4

        indices, centers = kmeans_quantize_nonzero(x, n_bits)
        x_hat = dequantize(indices, centers)

        # 原本为零的位置应该精确为零
        assert x_hat[0] == 0.0
        assert x_hat[1] == 0.0

    def test_quantize_then_dequantize_error_bound_small(self):
        """量化误差应该合理地小。"""
        torch.manual_seed(42)
        x = torch.randn(1000)
        n_bits = 4

        indices, centers = kmeans_quantize_nonzero(x, n_bits)
        x_hat = dequantize(indices, centers)

        # MSE 应该较小（相对于方差）
        mse = torch.mean((x - x_hat) ** 2)
        variance = torch.var(x)
        relative_error = mse / variance

        assert relative_error < 0.1  # 相对误差小于 10%

    def test_quantizer_deterministic_given_seed(self):
        """给定种子时量化应该是确定性的。"""
        x = torch.randn(100)
        n_bits = 4

        torch.manual_seed(123)
        indices1, centers1 = kmeans_quantize_nonzero(x, n_bits)

        torch.manual_seed(123)
        indices2, centers2 = kmeans_quantize_nonzero(x, n_bits)

        assert torch.equal(indices1, indices2)
        assert torch.allclose(centers1, centers2)

    def test_handles_all_nonzero_same_value(self):
        """应处理所有非零值相同的输入。"""
        x = torch.tensor([0.0, 5.0, 5.0, 5.0, 0.0])
        n_bits = 4

        indices, centers = kmeans_quantize_nonzero(x, n_bits)
        x_hat = dequantize(indices, centers)

        # 非零值应该都映射到同一个中心
        assert torch.allclose(x_hat[1:4], torch.tensor([5.0, 5.0, 5.0]))

    def test_handles_too_few_unique_values(self):
        """应处理唯一值少于 2^n - 1 的情况。"""
        x = torch.tensor([0.0, 1.0, 2.0, 1.0, 2.0])  # 只有 2 个唯一非零值
        n_bits = 4  # 通常会有 15 个中心

        indices, centers = kmeans_quantize_nonzero(x, n_bits)
        x_hat = dequantize(indices, centers)

        # 应该仍然工作，只是有效中心更少
        assert x_hat.shape == x.shape

    def test_separate_codebooks_for_weights_and_optimizer(self):
        """权重和优化器应该有独立的码本。"""
        weights = torch.randn(100)
        optimizer = torch.randn(100) * 10  # 不同的尺度
        n_bits = 4

        w_indices, w_centers = kmeans_quantize_nonzero(weights, n_bits)
        o_indices, o_centers = kmeans_quantize_nonzero(optimizer, n_bits)

        # 由于分布不同，中心应该不同
        # （这更多是使用模式测试）
        assert w_centers.shape == o_centers.shape


class TestInt4Packing:
    """int4 打包/解包测试。"""

    def test_pack_int4_roundtrip_even_length(self):
        """偶数长度的打包和解包应该是恒等操作。"""
        indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8)

        packed = pack_int4(indices)
        unpacked = unpack_int4(packed, len(indices))

        assert torch.equal(indices, unpacked)

    def test_pack_int4_roundtrip_odd_length(self):
        """奇数长度的打包和解包应该正常工作。"""
        indices = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)

        packed = pack_int4(indices)
        unpacked = unpack_int4(packed, len(indices))

        assert torch.equal(indices, unpacked)

    def test_pack_int4_compression_ratio(self):
        """打包后大小应约为原始大小的一半。"""
        indices = torch.randint(0, 16, (100,), dtype=torch.uint8)

        packed = pack_int4(indices)

        # 应该是 ceil(100/2) = 50 字节
        assert len(packed) == 50

    def test_pack_int4_preserves_values(self):
        """所有 0-15 的值应该被正确保留。"""
        indices = torch.arange(16, dtype=torch.uint8)

        packed = pack_int4(indices)
        unpacked = unpack_int4(packed, 16)

        assert torch.equal(indices, unpacked)


class TestQuantizationEdgeCases:
    """量化边界情况测试。"""

    def test_empty_tensor(self):
        """应处理空张量。"""
        x = torch.tensor([])
        n_bits = 4

        indices, centers = kmeans_quantize_nonzero(x, n_bits)
        x_hat = dequantize(indices, centers)

        assert x_hat.numel() == 0

    def test_all_zeros(self):
        """应处理全零张量。"""
        x = torch.zeros(100)
        n_bits = 4

        indices, centers = kmeans_quantize_nonzero(x, n_bits)
        x_hat = dequantize(indices, centers)

        assert torch.all(x_hat == 0)
        assert torch.all(indices == 0)

    def test_single_nonzero(self):
        """应处理只有单个非零值的张量。"""
        x = torch.tensor([0.0, 0.0, 5.0, 0.0])
        n_bits = 4

        indices, centers = kmeans_quantize_nonzero(x, n_bits)
        x_hat = dequantize(indices, centers)

        assert x_hat[2] != 0
        assert torch.allclose(x_hat[2], torch.tensor(5.0))
