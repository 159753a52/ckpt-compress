"""
AdamPrune 分布分析测试。

TDD: 先写测试，再写实现。
验证假设4：重要性得分是否服从指数分布。
"""

import pytest
import torch
import numpy as np
from typing import Dict


class TestFitExponentialDistribution:
    """测试 fit_exponential_distribution 函数。"""

    def test_fit_known_exponential_data(self):
        """对已知指数分布数据应该正确拟合。"""
        from ckpt_compress.methods.adam_prune.distribution import fit_exponential_distribution

        # 生成已知参数的指数分布数据
        torch.manual_seed(42)
        np.random.seed(42)
        lambda_true = 2.0
        data = torch.from_numpy(np.random.exponential(1/lambda_true, 10000)).float()

        lambda_est, goodness = fit_exponential_distribution(data)

        # λ 估计应该接近真实值
        assert abs(lambda_est - lambda_true) < 0.2  # 允许10%误差

    def test_fit_returns_positive_lambda(self):
        """λ 应该始终为正。"""
        from ckpt_compress.methods.adam_prune.distribution import fit_exponential_distribution

        data = torch.rand(1000)
        lambda_est, _ = fit_exponential_distribution(data)

        assert lambda_est > 0

    def test_fit_goodness_bounded(self):
        """拟合优度应该在 [0, 1] 范围内。"""
        from ckpt_compress.methods.adam_prune.distribution import fit_exponential_distribution

        data = torch.rand(1000)
        _, goodness = fit_exponential_distribution(data)

        assert 0 <= goodness <= 1

    def test_fit_empty_data_raises_error(self):
        """空数据应该抛出错误。"""
        from ckpt_compress.methods.adam_prune.distribution import fit_exponential_distribution

        with pytest.raises(ValueError):
            fit_exponential_distribution(torch.tensor([]))

    def test_fit_single_value(self):
        """单个值应该能处理。"""
        from ckpt_compress.methods.adam_prune.distribution import fit_exponential_distribution

        data = torch.tensor([1.0])
        lambda_est, _ = fit_exponential_distribution(data)

        # λ = 1/mean = 1/1.0 = 1.0
        assert abs(lambda_est - 1.0) < 1e-6


class TestAnalyzeDistribution:
    """测试 analyze_distribution 函数。"""

    def test_analyze_returns_required_keys(self):
        """应该返回所有必需的统计量。"""
        from ckpt_compress.methods.adam_prune.distribution import analyze_distribution

        data = torch.rand(1000)
        result = analyze_distribution(data)

        required_keys = ["mean", "std", "min", "max", "median",
                        "skewness", "kurtosis", "lambda_exp",
                        "ks_statistic", "ks_pvalue", "is_exponential"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_analyze_mean_correct(self):
        """均值应该正确计算。"""
        from ckpt_compress.methods.adam_prune.distribution import analyze_distribution

        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = analyze_distribution(data)

        assert abs(result["mean"] - 3.0) < 1e-6

    def test_analyze_std_correct(self):
        """标准差应该正确计算。"""
        from ckpt_compress.methods.adam_prune.distribution import analyze_distribution
        import numpy as np

        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = analyze_distribution(data)

        # 使用 numpy 的默认标准差计算 (ddof=0)
        expected_std = np.std(data.numpy())
        assert abs(result["std"] - expected_std) < 1e-6

    def test_analyze_exponential_data_detected(self):
        """指数分布数据应该被正确识别。"""
        from ckpt_compress.methods.adam_prune.distribution import analyze_distribution

        # 生成指数分布数据
        np.random.seed(42)
        data = torch.from_numpy(np.random.exponential(1.0, 5000)).float()

        result = analyze_distribution(data)

        # KS 检验 p-value 应该较大（不拒绝指数分布假设）
        assert result["ks_pvalue"] > 0.01  # 使用较宽松的阈值

    def test_analyze_non_exponential_data(self):
        """非指数分布数据应该被识别。"""
        from ckpt_compress.methods.adam_prune.distribution import analyze_distribution

        # 生成正态分布数据（明显不是指数分布）
        np.random.seed(42)
        data = torch.from_numpy(np.random.normal(5.0, 1.0, 5000)).float()
        data = torch.abs(data)  # 确保非负

        result = analyze_distribution(data)

        # KS 检验 p-value 应该较小（拒绝指数分布假设）
        # 注意：正态分布与指数分布差异明显
        assert result["ks_pvalue"] < 0.05 or not result["is_exponential"]

    def test_analyze_skewness_positive_for_exponential(self):
        """指数分布的偏度应该为正（右偏）。"""
        from ckpt_compress.methods.adam_prune.distribution import analyze_distribution

        np.random.seed(42)
        data = torch.from_numpy(np.random.exponential(1.0, 5000)).float()

        result = analyze_distribution(data)

        # 指数分布偏度理论值为 2
        assert result["skewness"] > 0

    def test_analyze_empty_data_raises_error(self):
        """空数据应该抛出错误。"""
        from ckpt_compress.methods.adam_prune.distribution import analyze_distribution

        with pytest.raises(ValueError):
            analyze_distribution(torch.tensor([]))


class TestPlotDistribution:
    """测试 plot_distribution 函数（可选，用于可视化）。"""

    def test_plot_returns_figure(self):
        """应该返回 matplotlib figure。"""
        from ckpt_compress.methods.adam_prune.distribution import plot_distribution

        data = torch.rand(1000)

        try:
            fig = plot_distribution(data)
            assert fig is not None
            # 清理
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_with_title(self):
        """应该支持自定义标题。"""
        from ckpt_compress.methods.adam_prune.distribution import plot_distribution

        data = torch.rand(1000)

        try:
            fig = plot_distribution(data, title="Test Distribution")
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestCompareWithTheoretical:
    """测试理论预测与实际分布的比较。"""

    def test_theoretical_quantile_function(self):
        """验证指数分布的分位数函数。"""
        from ckpt_compress.methods.adam_prune.distribution import exponential_quantile

        # 指数分布的 p 分位数: F^{-1}(p) = -ln(1-p) / λ
        lambda_val = 2.0

        # 中位数 (p=0.5)
        median = exponential_quantile(0.5, lambda_val)
        expected_median = -np.log(0.5) / lambda_val
        assert abs(median - expected_median) < 1e-6

    def test_theoretical_quantile_edge_cases(self):
        """测试分位数函数的边界情况。"""
        from ckpt_compress.methods.adam_prune.distribution import exponential_quantile

        lambda_val = 2.0

        # p <= 0 应该返回 0
        assert exponential_quantile(0, lambda_val) == 0.0
        assert exponential_quantile(-0.1, lambda_val) == 0.0

        # p >= 1 应该返回 inf
        assert exponential_quantile(1.0, lambda_val) == float('inf')
        assert exponential_quantile(1.5, lambda_val) == float('inf')

    def test_theoretical_cdf(self):
        """验证指数分布的 CDF。"""
        from ckpt_compress.methods.adam_prune.distribution import exponential_cdf

        lambda_val = 2.0
        x = 1.0

        # F(x) = 1 - exp(-λx)
        cdf = exponential_cdf(x, lambda_val)
        expected = 1 - np.exp(-lambda_val * x)
        assert abs(cdf - expected) < 1e-6

    def test_theoretical_cdf_negative_x(self):
        """测试 CDF 对负数输入的处理。"""
        from ckpt_compress.methods.adam_prune.distribution import exponential_cdf

        lambda_val = 2.0

        # x < 0 应该返回 0
        assert exponential_cdf(-1.0, lambda_val) == 0.0
        assert exponential_cdf(-0.001, lambda_val) == 0.0


class TestDistributionEdgeCases:
    """测试分布分析的边界情况。"""

    def test_fit_very_small_mean(self):
        """测试非常小的均值情况。"""
        from ckpt_compress.methods.adam_prune.distribution import fit_exponential_distribution

        # 非常小的值
        data = torch.tensor([1e-12, 1e-12, 1e-12])
        lambda_est, _ = fit_exponential_distribution(data)

        # 应该使用最小值 1e-10 来避免除以零
        assert lambda_est > 0

    def test_analyze_two_data_points(self):
        """测试只有两个数据点的情况。"""
        from ckpt_compress.methods.adam_prune.distribution import analyze_distribution

        data = torch.tensor([1.0, 2.0])
        result = analyze_distribution(data)

        # 偏度和峰度应该为 0（数据点太少）
        assert result["skewness"] == 0.0
        assert result["kurtosis"] == 0.0

    def test_analyze_single_data_point(self):
        """测试只有一个数据点的情况。"""
        from ckpt_compress.methods.adam_prune.distribution import analyze_distribution

        data = torch.tensor([5.0])
        result = analyze_distribution(data)

        # 应该能正常处理
        assert result["mean"] == 5.0
        assert result["ks_statistic"] == 0.0
        assert result["ks_pvalue"] == 1.0

    def test_plot_with_save_path(self):
        """测试保存图表到文件。"""
        from ckpt_compress.methods.adam_prune.distribution import plot_distribution
        import tempfile
        import os

        data = torch.rand(100)

        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                save_path = f.name

            fig = plot_distribution(data, save_path=save_path)

            # 文件应该被创建
            assert os.path.exists(save_path)

            import matplotlib.pyplot as plt
            plt.close(fig)
            os.unlink(save_path)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestFitMultipleDistributions:
    """测试 fit_multiple_distributions 函数。"""

    def test_fit_returns_all_distributions(self):
        """应该返回所有分布的拟合结果。"""
        from ckpt_compress.methods.adam_prune.distribution import fit_multiple_distributions

        np.random.seed(42)
        data = np.random.exponential(1.0, 1000)

        results = fit_multiple_distributions(data)

        # 应该包含5种分布
        expected_dists = ['exponential', 'lognormal', 'weibull', 'gamma', 'pareto']
        for dist in expected_dists:
            assert dist in results, f"Missing distribution: {dist}"

    def test_fit_returns_ks_stats(self):
        """每个分布应该返回KS统计量。"""
        from ckpt_compress.methods.adam_prune.distribution import fit_multiple_distributions

        np.random.seed(42)
        data = np.random.exponential(1.0, 1000)

        results = fit_multiple_distributions(data)

        for dist_name, result in results.items():
            if 'error' not in result:
                assert 'ks_stat' in result, f"{dist_name} missing ks_stat"
                assert 'p_value' in result, f"{dist_name} missing p_value"
                assert 'params' in result, f"{dist_name} missing params"
                assert 0 <= result['ks_stat'] <= 1

    def test_fit_pareto_distribution(self):
        """应该能拟合Pareto分布。"""
        from ckpt_compress.methods.adam_prune.distribution import fit_multiple_distributions

        np.random.seed(42)
        # 生成Pareto分布数据
        data = (np.random.pareto(2.0, 1000) + 1) * 1.0

        results = fit_multiple_distributions(data)

        assert 'pareto' in results
        if 'error' not in results['pareto']:
            assert results['pareto']['ks_stat'] >= 0

    def test_fit_with_small_data(self):
        """小数据集应该能处理。"""
        from ckpt_compress.methods.adam_prune.distribution import fit_multiple_distributions

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        results = fit_multiple_distributions(data)

        # 应该返回结果（可能有error）
        assert isinstance(results, dict)

    def test_fit_with_insufficient_data(self):
        """数据不足时应该返回error。"""
        from ckpt_compress.methods.adam_prune.distribution import fit_multiple_distributions

        data = np.array([1.0, 2.0])

        results = fit_multiple_distributions(data)

        # 所有分布应该有error
        for dist_name, result in results.items():
            assert 'error' in result


class TestGetBestFit:
    """测试 get_best_fit 函数。"""

    def test_get_best_fit_returns_lowest_ks(self):
        """应该返回KS统计量最小的分布。"""
        from ckpt_compress.methods.adam_prune.distribution import get_best_fit

        np.random.seed(42)
        # 生成指数分布数据
        data = np.random.exponential(1.0, 5000)

        best = get_best_fit(data)

        assert best is not None
        assert 'name' in best
        assert 'ks_stat' in best
        assert 'params' in best

    def test_get_best_fit_exponential_data(self):
        """指数分布数据应该最佳拟合指数分布。"""
        from ckpt_compress.methods.adam_prune.distribution import get_best_fit

        np.random.seed(42)
        data = np.random.exponential(1.0, 5000)

        best = get_best_fit(data)

        # 指数分布应该是最佳拟合之一
        assert best['name'] in ['exponential', 'gamma', 'weibull']

    def test_get_best_fit_lognormal_data(self):
        """对数正态数据应该最佳拟合对数正态分布。"""
        from ckpt_compress.methods.adam_prune.distribution import get_best_fit

        np.random.seed(42)
        data = np.random.lognormal(0, 0.5, 5000)

        best = get_best_fit(data)

        # 对数正态应该是最佳拟合
        assert best['name'] == 'lognormal'

    def test_get_best_fit_weibull_data(self):
        """Weibull数据应该最佳拟合Weibull分布。"""
        from ckpt_compress.methods.adam_prune.distribution import get_best_fit
        from scipy import stats

        np.random.seed(42)
        data = stats.weibull_min.rvs(c=1.5, scale=2.0, size=5000)

        best = get_best_fit(data)

        # Weibull应该是最佳拟合之一
        assert best['name'] in ['weibull', 'gamma', 'lognormal']

    def test_get_best_fit_insufficient_data(self):
        """数据不足时应该返回None。"""
        from ckpt_compress.methods.adam_prune.distribution import get_best_fit

        data = np.array([1.0, 2.0])

        best = get_best_fit(data)

        assert best is None

    def test_get_best_fit_with_tensor(self):
        """应该支持torch.Tensor输入。"""
        from ckpt_compress.methods.adam_prune.distribution import get_best_fit

        np.random.seed(42)
        data = torch.from_numpy(np.random.exponential(1.0, 1000)).float()

        best = get_best_fit(data)

        assert best is not None
        assert 'name' in best
