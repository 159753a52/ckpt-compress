"""
快速分布拟合模块测试 (TDD)。

测试 GPU 友好的韦伯分布线性回归拟合和快速 KS 检验。
"""

import pytest
import torch
import numpy as np
from typing import Dict
from scipy import stats


# ============================================================
# 测试 fit_weibull_linear_regression (韦伯分布线性回归拟合)
# ============================================================
class TestFitWeibullLinearRegression:
    """测试韦伯分布线性回归拟合函数。"""

    def test_fit_known_weibull_data(self):
        """对已知韦伯分布数据应该正确拟合参数。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import fit_weibull_linear_regression

        # 生成已知参数的韦伯分布数据
        np.random.seed(42)
        k_true = 1.5  # 形状参数
        lam_true = 2.0  # 尺度参数
        data = torch.from_numpy(
            stats.weibull_min.rvs(c=k_true, scale=lam_true, size=10000)
        ).float()

        k_est, lam_est = fit_weibull_linear_regression(data)

        # 允许 20% 误差（线性回归是近似方法）
        assert abs(k_est - k_true) / k_true < 0.2, f"k_est={k_est}, k_true={k_true}"
        assert abs(lam_est - lam_true) / lam_true < 0.2, f"lam_est={lam_est}, lam_true={lam_true}"

    def test_fit_returns_positive_parameters(self):
        """形状参数 k 和尺度参数 λ 应该始终为正。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import fit_weibull_linear_regression

        torch.manual_seed(42)
        data = torch.rand(1000) + 0.01  # 确保正值

        k, lam = fit_weibull_linear_regression(data)

        assert k > 0, f"k should be positive, got {k}"
        assert lam > 0, f"lam should be positive, got {lam}"

    def test_fit_exponential_as_special_case(self):
        """指数分布是 k=1 的韦伯分布特例。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import fit_weibull_linear_regression

        np.random.seed(42)
        # 指数分布 = 韦伯分布 (k=1)
        data = torch.from_numpy(np.random.exponential(2.0, 10000)).float()

        k_est, lam_est = fit_weibull_linear_regression(data)

        # k 应该接近 1
        assert abs(k_est - 1.0) < 0.3, f"k_est={k_est}, expected ~1.0"

    def test_fit_with_gpu_tensor(self):
        """应该支持 GPU tensor（如果可用）。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import fit_weibull_linear_regression

        np.random.seed(42)
        data = torch.from_numpy(
            stats.weibull_min.rvs(c=1.5, scale=2.0, size=1000)
        ).float()

        if torch.cuda.is_available():
            data = data.cuda()

        k, lam = fit_weibull_linear_regression(data)

        assert k > 0
        assert lam > 0

    def test_fit_small_data(self):
        """小数据集应该能处理。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import fit_weibull_linear_regression

        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        k, lam = fit_weibull_linear_regression(data)

        assert k > 0
        assert lam > 0

    def test_fit_large_data_performance(self):
        """大数据集应该能快速处理。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import fit_weibull_linear_regression
        import time

        np.random.seed(42)
        data = torch.from_numpy(
            stats.weibull_min.rvs(c=1.5, scale=2.0, size=1000000)
        ).float()

        start = time.time()
        k, lam = fit_weibull_linear_regression(data)
        elapsed = time.time() - start

        # 应该在 1 秒内完成（主要是排序时间）
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s"
        assert k > 0
        assert lam > 0

    def test_fit_with_zeros_filtered(self):
        """包含零值的数据应该被正确过滤。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import fit_weibull_linear_regression

        data = torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        k, lam = fit_weibull_linear_regression(data)

        assert k > 0
        assert lam > 0


# ============================================================
# 测试 ks_test_gpu (GPU 上的 KS 检验)
# ============================================================
class TestKsTestGpu:
    """测试 GPU 上的 Kolmogorov-Smirnov 检验。"""

    def test_ks_perfect_fit(self):
        """完美拟合的数据应该有很小的 KS 统计量。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            ks_test_gpu,
            weibull_cdf_gpu,
        )

        np.random.seed(42)
        k, lam = 1.5, 2.0
        data = torch.from_numpy(
            stats.weibull_min.rvs(c=k, scale=lam, size=5000)
        ).float()

        ks_stat = ks_test_gpu(data, lambda x: weibull_cdf_gpu(x, k, lam))

        # KS 统计量应该较小（好的拟合）
        assert ks_stat < 0.05, f"KS stat too large: {ks_stat}"

    def test_ks_poor_fit(self):
        """差的拟合应该有较大的 KS 统计量。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            ks_test_gpu,
            weibull_cdf_gpu,
        )

        np.random.seed(42)
        # 生成正态分布数据，但用韦伯分布拟合
        data = torch.from_numpy(np.abs(np.random.normal(5.0, 1.0, 5000))).float()

        ks_stat = ks_test_gpu(data, lambda x: weibull_cdf_gpu(x, 1.5, 2.0))

        # KS 统计量应该较大（差的拟合）
        assert ks_stat > 0.1, f"KS stat too small for poor fit: {ks_stat}"

    def test_ks_returns_bounded_value(self):
        """KS 统计量应该在 [0, 1] 范围内。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            ks_test_gpu,
            weibull_cdf_gpu,
        )

        data = torch.rand(1000) + 0.01

        ks_stat = ks_test_gpu(data, lambda x: weibull_cdf_gpu(x, 1.0, 1.0))

        assert 0 <= ks_stat <= 1, f"KS stat out of bounds: {ks_stat}"

    def test_ks_with_gpu_tensor(self):
        """应该支持 GPU tensor。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            ks_test_gpu,
            weibull_cdf_gpu,
        )

        data = torch.rand(1000) + 0.01

        if torch.cuda.is_available():
            data = data.cuda()

        ks_stat = ks_test_gpu(data, lambda x: weibull_cdf_gpu(x, 1.0, 1.0))

        assert 0 <= ks_stat <= 1

    def test_ks_matches_scipy(self):
        """结果应该与 scipy.stats.kstest 接近。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            ks_test_gpu,
            weibull_cdf_gpu,
        )

        np.random.seed(42)
        k, lam = 1.5, 2.0
        data_np = stats.weibull_min.rvs(c=k, scale=lam, size=1000)
        data = torch.from_numpy(data_np).float()

        # GPU 版本
        ks_gpu = ks_test_gpu(data, lambda x: weibull_cdf_gpu(x, k, lam))

        # scipy 版本
        ks_scipy, _ = stats.kstest(data_np, stats.weibull_min.cdf, args=(k, 0, lam))

        # 允许 10% 误差
        assert abs(ks_gpu - ks_scipy) < 0.1, f"GPU: {ks_gpu}, scipy: {ks_scipy}"


# ============================================================
# 测试 CDF 函数
# ============================================================
class TestCdfFunctions:
    """测试 CDF 函数。"""

    def test_weibull_cdf_at_zero(self):
        """x=0 时 CDF 应该为 0。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import weibull_cdf_gpu

        x = torch.tensor([0.0])
        cdf = weibull_cdf_gpu(x, k=1.5, lam=2.0)

        assert cdf.item() == 0.0

    def test_weibull_cdf_at_infinity(self):
        """x 很大时 CDF 应该接近 1。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import weibull_cdf_gpu

        x = torch.tensor([1000.0])
        cdf = weibull_cdf_gpu(x, k=1.5, lam=2.0)

        assert cdf.item() > 0.999

    def test_weibull_cdf_monotonic(self):
        """CDF 应该单调递增。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import weibull_cdf_gpu

        x = torch.linspace(0.01, 10, 100)
        cdf = weibull_cdf_gpu(x, k=1.5, lam=2.0)

        # 检查单调性
        diff = cdf[1:] - cdf[:-1]
        assert (diff >= 0).all(), "CDF should be monotonically increasing"

    def test_weibull_cdf_matches_scipy(self):
        """应该与 scipy 的 CDF 匹配。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import weibull_cdf_gpu

        k, lam = 1.5, 2.0
        x = torch.tensor([0.5, 1.0, 2.0, 3.0, 5.0])

        cdf_gpu = weibull_cdf_gpu(x, k, lam)
        cdf_scipy = stats.weibull_min.cdf(x.numpy(), c=k, scale=lam)

        np.testing.assert_allclose(cdf_gpu.numpy(), cdf_scipy, rtol=1e-5)

    def test_exponential_cdf_gpu(self):
        """测试指数分布 CDF。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import exponential_cdf_gpu

        lam = 2.0
        x = torch.tensor([0.0, 0.5, 1.0, 2.0])

        cdf = exponential_cdf_gpu(x, lam)
        expected = 1 - torch.exp(-lam * x)

        torch.testing.assert_close(cdf, expected)


# ============================================================
# 测试 fit_distributions_per_layer_fast
# ============================================================
class TestFitDistributionsPerLayerFast:
    """测试快速分布拟合函数。"""

    def test_fit_single_layer(self):
        """应该能拟合单层数据。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_distributions_per_layer_fast,
        )

        np.random.seed(42)
        layer_scores = {
            "layer1.weight": torch.from_numpy(
                stats.weibull_min.rvs(c=1.5, scale=2.0, size=1000)
            ).float()
        }

        results = fit_distributions_per_layer_fast(layer_scores)

        assert "layer1.weight" in results
        assert "name" in results["layer1.weight"]
        assert "params" in results["layer1.weight"]
        assert results["layer1.weight"]["name"] == "weibull"

    def test_fit_multiple_layers(self):
        """应该能拟合多层数据。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_distributions_per_layer_fast,
        )

        np.random.seed(42)
        layer_scores = {
            "layer1.weight": torch.rand(1000) + 0.01,
            "layer2.weight": torch.rand(2000) + 0.01,
            "layer3.weight": torch.rand(500) + 0.01,
        }

        results = fit_distributions_per_layer_fast(layer_scores)

        assert len(results) == 3
        for name in layer_scores.keys():
            assert name in results

    def test_fit_with_downsampling(self):
        """大数据应该被下采样。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_distributions_per_layer_fast,
        )

        np.random.seed(42)
        # 大数据
        layer_scores = {
            "large_layer": torch.rand(1000000) + 0.01,
        }

        results = fit_distributions_per_layer_fast(layer_scores, max_samples=10000)

        assert "large_layer" in results
        assert results["large_layer"]["params"] is not None

    def test_fit_returns_scipy_format_params(self):
        """参数应该是 scipy 格式 (c, loc, scale)。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_distributions_per_layer_fast,
        )

        layer_scores = {"layer": torch.rand(1000) + 0.01}

        results = fit_distributions_per_layer_fast(layer_scores)

        params = results["layer"]["params"]
        assert len(params) == 3, f"Expected 3 params (c, loc, scale), got {len(params)}"

    def test_fit_with_gpu_tensors(self):
        """应该支持 GPU tensor。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_distributions_per_layer_fast,
        )

        layer_scores = {"layer": torch.rand(1000) + 0.01}

        if torch.cuda.is_available():
            layer_scores = {k: v.cuda() for k, v in layer_scores.items()}

        results = fit_distributions_per_layer_fast(layer_scores)

        assert "layer" in results

    def test_fit_performance(self):
        """150 层应该在几秒内完成。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_distributions_per_layer_fast,
        )
        import time

        np.random.seed(42)
        # 模拟 GPT-2 的层数和参数量
        layer_scores = {}
        for i in range(150):
            size = np.random.randint(1000, 100000)
            layer_scores[f"layer{i}"] = torch.rand(size) + 0.01

        start = time.time()
        results = fit_distributions_per_layer_fast(layer_scores, max_samples=10000)
        elapsed = time.time() - start

        assert len(results) == 150
        # 应该在 30 秒内完成（保守估计，CPU 上可能较慢）
        assert elapsed < 30.0, f"Too slow: {elapsed:.2f}s for 150 layers"

    def test_fit_empty_dict(self):
        """空字典应该返回空结果。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_distributions_per_layer_fast,
        )

        results = fit_distributions_per_layer_fast({})

        assert results == {}

    def test_fit_with_negative_values(self):
        """包含负值的数据应该被过滤。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_distributions_per_layer_fast,
        )

        # 包含负值
        layer_scores = {"layer": torch.randn(1000)}  # 正态分布，有负值

        results = fit_distributions_per_layer_fast(layer_scores)

        assert "layer" in results
        # 应该能处理（过滤负值后拟合）

    def test_fit_returns_ks_stat(self):
        """应该返回 KS 统计量。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_distributions_per_layer_fast,
        )

        layer_scores = {"layer": torch.rand(1000) + 0.01}

        results = fit_distributions_per_layer_fast(layer_scores)

        assert "ks_stat" in results["layer"]
        assert 0 <= results["layer"]["ks_stat"] <= 1


# ============================================================
# 测试与原有接口的兼容性
# ============================================================
class TestBackwardCompatibility:
    """测试与原有 fit_distributions_per_layer 的兼容性。"""

    def test_output_format_compatible(self):
        """输出格式应该与原函数兼容。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_distributions_per_layer_fast,
        )

        layer_scores = {"layer": torch.rand(1000) + 0.01}

        results = fit_distributions_per_layer_fast(layer_scores)

        # 检查必需的字段
        result = results["layer"]
        assert "name" in result
        assert "params" in result
        assert "ks_stat" in result

    def test_can_be_used_with_build_layer_dist_params(self):
        """应该能与 build_layer_dist_params 配合使用。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_distributions_per_layer_fast,
        )
        from ckpt_compress.methods.adam_prune.adaptive_pruning import (
            build_layer_dist_params,
        )

        layer_scores = {
            "layer1": torch.rand(1000) + 0.01,
            "layer2": torch.rand(2000) + 0.01,
        }

        fit_results = fit_distributions_per_layer_fast(layer_scores)

        # 应该能正常调用
        layer_dist_params = build_layer_dist_params(layer_scores, fit_results)

        assert "layer1" in layer_dist_params
        assert "layer2" in layer_dist_params
        assert "N" in layer_dist_params["layer1"]


# ============================================================
# 测试边界情况
# ============================================================
class TestEdgeCases:
    """测试边界情况。"""

    def test_all_same_values(self):
        """所有值相同时应该能处理。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_weibull_linear_regression,
        )

        data = torch.ones(100)

        # 应该不会崩溃，但结果可能不准确
        try:
            k, lam = fit_weibull_linear_regression(data)
            # 只要不崩溃就行
            assert True
        except Exception as e:
            # 如果抛出异常，应该是合理的异常
            assert "constant" in str(e).lower() or "zero" in str(e).lower()

    def test_very_small_values(self):
        """非常小的值应该能处理。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_weibull_linear_regression,
        )

        data = torch.rand(1000) * 1e-10 + 1e-12

        k, lam = fit_weibull_linear_regression(data)

        assert k > 0
        assert lam > 0

    def test_very_large_values(self):
        """非常大的值应该能处理。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_weibull_linear_regression,
        )

        data = torch.rand(1000) * 1e10 + 1e8

        k, lam = fit_weibull_linear_regression(data)

        assert k > 0
        assert lam > 0

    def test_mixed_dtypes(self):
        """不同数据类型应该能处理。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_weibull_linear_regression,
        )

        # float64
        data64 = torch.rand(100, dtype=torch.float64) + 0.01
        k64, lam64 = fit_weibull_linear_regression(data64)
        assert k64 > 0

        # float32
        data32 = torch.rand(100, dtype=torch.float32) + 0.01
        k32, lam32 = fit_weibull_linear_regression(data32)
        assert k32 > 0

    def test_minimum_data_points(self):
        """最少数据点应该能处理。"""
        from ckpt_compress.methods.adam_prune.distribution_fast import (
            fit_weibull_linear_regression,
        )

        # 至少需要 2 个点做线性回归
        data = torch.tensor([1.0, 2.0])

        k, lam = fit_weibull_linear_regression(data)

        assert k > 0
        assert lam > 0
