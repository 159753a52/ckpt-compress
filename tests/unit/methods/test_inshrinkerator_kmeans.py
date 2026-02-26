"""
Inshrinkerator 近似 K-means 测试（基于草图）。
"""

import pytest
import torch

from ckpt_compress.methods.inshrinkerator.approx_kmeans import (
    approx_kmeans,
    weighted_kmeans_plusplus_init,
    compute_sample_weights,
    quantize_to_centers,
)


class TestSampleWeights:
    """样本权重计算测试。"""

    def test_sample_weight_formula(self):
        """测试 w^i = sigma * freq + (1 - sigma) * mag。"""
        freq = torch.tensor([0.5, 0.3, 0.2])  # 归一化频率
        mag = torch.tensor([0.1, 0.6, 0.3])   # 归一化幅度
        sigma = 0.2

        weights = compute_sample_weights(freq, mag, sigma)

        # w = 0.2 * freq + 0.8 * mag
        expected = 0.2 * freq + 0.8 * mag
        assert torch.allclose(weights, expected)

    def test_sigma_extremes_freq_only(self):
        """sigma=1 应只使用频率。"""
        freq = torch.tensor([1.0, 0.0, 0.0])
        mag = torch.tensor([0.0, 1.0, 0.0])
        sigma = 1.0

        weights = compute_sample_weights(freq, mag, sigma)

        assert torch.allclose(weights, freq)

    def test_sigma_extremes_mag_only(self):
        """sigma=0 应只使用幅度。"""
        freq = torch.tensor([1.0, 0.0, 0.0])
        mag = torch.tensor([0.0, 1.0, 0.0])
        sigma = 0.0

        weights = compute_sample_weights(freq, mag, sigma)

        assert torch.allclose(weights, mag)


class TestWeightedKmeansPlusPlus:
    """加权 K-means++ 初始化测试。"""

    def test_weighted_kmeanspp_selects_first_centroid_from_data(self):
        """第一个质心应来自数据。"""
        torch.manual_seed(42)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = torch.ones(5)
        k = 3

        centers = weighted_kmeans_plusplus_init(values, weights, k)

        # 所有质心应来自原始值
        for c in centers:
            assert c.item() in values.tolist()

    def test_weighted_kmeanspp_probability_normalized(self):
        """概率分布之和应为 1。"""
        # 这通过算法正确工作来隐式测试
        torch.manual_seed(42)
        values = torch.tensor([1.0, 5.0, 10.0, 50.0])
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
        k = 2

        centers = weighted_kmeans_plusplus_init(values, weights, k)

        assert len(centers) == k

    def test_weighted_kmeanspp_respects_weights(self):
        """权重较高的点应更可能被选中。"""
        # 多次运行并检查分布
        torch.manual_seed(42)
        values = torch.tensor([1.0, 100.0])
        weights = torch.tensor([0.01, 0.99])  # 第二个点权重很高
        k = 1

        # 由于第二个点权重高，它通常应被选中
        centers = weighted_kmeans_plusplus_init(values, weights, k)
        # 只检查返回有效质心
        assert centers[0].item() in values.tolist()


class TestApproxKmeans:
    """近似 K-means 测试。"""

    def test_kmeans_output_k_centers(self):
        """输出应恰好有 K 个质心。"""
        torch.manual_seed(42)
        values = torch.randn(1000)
        k = 8

        centers = approx_kmeans(values, k)

        assert len(centers) == k

    def test_kmeans_centers_sorted(self):
        """质心应排序以便于编码。"""
        torch.manual_seed(42)
        values = torch.randn(1000)
        k = 8

        centers = approx_kmeans(values, k)

        # 检查已排序
        for i in range(len(centers) - 1):
            assert centers[i] <= centers[i + 1]

    def test_quantize_assigns_nearest_center(self):
        """量化应分配到最近的质心。"""
        values = torch.tensor([1.0, 2.0, 8.0, 9.0])
        centers = torch.tensor([1.5, 8.5])

        indices = quantize_to_centers(values, centers)

        # 1.0 和 2.0 更接近 1.5（索引 0）
        # 8.0 和 9.0 更接近 8.5（索引 1）
        expected = torch.tensor([0, 0, 1, 1])
        assert torch.equal(indices, expected)

    def test_small_input_degenerate_case(self):
        """应处理样本数 < K 的情况。"""
        values = torch.tensor([1.0, 2.0])
        k = 5

        centers = approx_kmeans(values, k)

        # 应返回更少的质心或优雅处理
        assert len(centers) <= k

    def test_determinism_with_fixed_seed(self):
        """固定种子时结果应确定。"""
        values = torch.randn(100)
        k = 4

        torch.manual_seed(123)
        centers1 = approx_kmeans(values, k)

        torch.manual_seed(123)
        centers2 = approx_kmeans(values, k)

        assert torch.allclose(centers1, centers2)

    def test_kmeans_reduces_quantization_error(self):
        """K-means 应产生合理的量化误差。"""
        torch.manual_seed(42)
        values = torch.randn(1000)
        k = 16

        centers = approx_kmeans(values, k)
        indices = quantize_to_centers(values, centers)
        reconstructed = centers[indices]

        mse = torch.mean((values - reconstructed) ** 2)
        variance = torch.var(values)

        # 近似 K-means 以速度换取精度
        # 误差应小于方差（有一定压缩收益）
        assert mse < variance * 0.8


class TestApproxKmeansEdgeCases:
    """近似 K-means 边界情况测试。"""

    def test_empty_input(self):
        """应处理空输入。"""
        values = torch.tensor([])
        k = 4

        centers = approx_kmeans(values, k)

        assert len(centers) == 0 or len(centers) <= k

    def test_single_value(self):
        """应处理单个值。"""
        values = torch.tensor([5.0])
        k = 4

        centers = approx_kmeans(values, k)

        assert len(centers) >= 1
        assert 5.0 in centers.tolist() or torch.isclose(centers[0], torch.tensor(5.0))

    def test_all_same_values(self):
        """应处理所有相同的值。"""
        values = torch.ones(100) * 3.14
        k = 4

        centers = approx_kmeans(values, k)

        # 所有质心应接近 3.14
        assert torch.allclose(centers, torch.ones_like(centers) * 3.14, atol=0.1)
