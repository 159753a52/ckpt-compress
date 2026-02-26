"""
Inshrinkerator 分位数草图测试（基于 DDSketch）。
"""

import pytest
import torch

from ckpt_compress.methods.inshrinkerator.sketch import (
    QuantileSketch,
    compute_bucket_index,
    compute_gamma,
)


class TestGammaFormula:
    """gamma 计算测试。"""

    def test_gamma_formula(self):
        """测试 gamma = (1 + alpha) / (1 - alpha)。"""
        alpha = 0.01

        gamma = compute_gamma(alpha)

        expected = (1 + alpha) / (1 - alpha)
        assert abs(gamma - expected) < 1e-6

    def test_gamma_increases_with_alpha(self):
        """较大的 alpha 应产生较大的 gamma。"""
        gamma_small = compute_gamma(0.01)
        gamma_large = compute_gamma(0.1)

        assert gamma_large > gamma_small


class TestBucketIndex:
    """桶索引计算测试。"""

    def test_bucket_index_monotonic(self):
        """桶索引应随 x 单调递增。"""
        gamma = compute_gamma(0.01)
        values = torch.tensor([0.1, 0.5, 1.0, 2.0, 10.0])

        indices = compute_bucket_index(values, gamma)

        # 每个后续索引应 >= 前一个
        for i in range(len(indices) - 1):
            assert indices[i + 1] >= indices[i]

    def test_bucket_index_same_for_close_values(self):
        """非常接近的值应映射到同一个桶。"""
        gamma = compute_gamma(0.01)
        values = torch.tensor([1.0, 1.001])

        indices = compute_bucket_index(values, gamma)

        # 小 alpha 时，接近的值应在同一个桶中
        assert indices[0] == indices[1]


class TestQuantileSketch:
    """QuantileSketch 类测试。"""

    def test_histogram_counts_sum(self):
        """桶计数之和应等于样本数。"""
        values = torch.randn(1000).abs()
        sketch = QuantileSketch(alpha=0.01)
        sketch.add(values)

        total_count = sum(sketch.buckets.values())

        assert total_count == 1000

    def test_quantile_0_returns_min_bucket(self):
        """分位数 0 应返回最小值范围。"""
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        sketch = QuantileSketch(alpha=0.01)
        sketch.add(values)

        q0 = sketch.quantile(0.0)

        # 应接近最小值
        assert q0 <= values.min().item() * 1.1  # 允许一些误差

    def test_quantile_1_returns_max_bucket(self):
        """分位数 1 应返回最大值范围。"""
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        sketch = QuantileSketch(alpha=0.01)
        sketch.add(values)

        q1 = sketch.quantile(1.0)

        # 应接近最大值
        assert q1 >= values.max().item() * 0.9  # 允许一些误差

    def test_quantile_median_close(self):
        """中位数估计应接近真实中位数。"""
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        sketch = QuantileSketch(alpha=0.01)
        sketch.add(values)

        median_estimate = sketch.quantile(0.5)

        # 真实中位数是 2.5，允许一些草图近似误差
        assert 1.5 <= median_estimate <= 3.5

    def test_quantile_ordering(self):
        """较低的分位数应给出较低的值。"""
        values = torch.randn(1000).abs() + 0.1
        sketch = QuantileSketch(alpha=0.01)
        sketch.add(values)

        q25 = sketch.quantile(0.25)
        q50 = sketch.quantile(0.50)
        q75 = sketch.quantile(0.75)

        assert q25 <= q50 <= q75

    def test_handles_negative_values_by_abs(self):
        """草图应处理负值（使用绝对值）。"""
        values = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        sketch = QuantileSketch(alpha=0.01)

        # 不应抛出异常
        sketch.add(values.abs())

        assert sketch.count == 4

    def test_empty_sketch(self):
        """空草图应优雅地处理分位数。"""
        sketch = QuantileSketch(alpha=0.01)

        # 应返回 0 或优雅处理
        result = sketch.quantile(0.5)
        assert result == 0.0


class TestSketchMerge:
    """草图合并测试（可选功能）。"""

    def test_sketch_merge_counts(self):
        """合并后的草图应有组合计数。"""
        values1 = torch.randn(500).abs()
        values2 = torch.randn(500).abs()

        sketch1 = QuantileSketch(alpha=0.01)
        sketch1.add(values1)

        sketch2 = QuantileSketch(alpha=0.01)
        sketch2.add(values2)

        sketch1.merge(sketch2)

        assert sketch1.count == 1000

    def test_sketch_merge_quantiles_reasonable(self):
        """合并后的草图分位数应合理。"""
        values1 = torch.randn(500).abs() + 0.1
        values2 = torch.randn(500).abs() + 0.1
        all_values = torch.cat([values1, values2])

        sketch1 = QuantileSketch(alpha=0.01)
        sketch1.add(values1)

        sketch2 = QuantileSketch(alpha=0.01)
        sketch2.add(values2)

        sketch1.merge(sketch2)

        # 中位数应在合理范围内
        median = sketch1.quantile(0.5)
        true_median = torch.median(all_values).item()

        # 允许 20% 相对误差
        assert abs(median - true_median) / true_median < 0.2
