"""
AdamPrune 分布可视化测试。

TDD: 先写测试，再写实现。
"""

import pytest
import torch
import numpy as np
from typing import Dict, List


class TestPlotLayerDistributions:
    """测试 plot_layer_distributions 函数。"""

    def test_returns_figure(self):
        """应该返回 matplotlib figure。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_distributions

        # 创建测试数据
        layer_scores = {
            'layer1': np.random.exponential(1.0, 1000),
            'layer2': np.random.lognormal(0, 1, 1000),
        }

        try:
            fig = plot_layer_distributions(layer_scores)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_handles_single_layer(self):
        """应该能处理单个层。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_distributions

        layer_scores = {
            'single_layer': np.random.exponential(1.0, 1000),
        }

        try:
            fig = plot_layer_distributions(layer_scores)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_handles_multiple_layers(self):
        """应该能处理多个层。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_distributions

        layer_scores = {
            'attention': np.random.exponential(1.0, 1000),
            'mlp': np.random.lognormal(0, 1, 1000),
            'embedding': np.random.weibull(1.5, 1000),
            'layernorm': np.random.gamma(2, 1, 1000),
        }

        try:
            fig = plot_layer_distributions(layer_scores)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_with_custom_title(self):
        """应该支持自定义标题。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_distributions

        layer_scores = {
            'layer1': np.random.exponential(1.0, 1000),
        }

        try:
            fig = plot_layer_distributions(layer_scores, title="Custom Title")
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_save_to_file(self):
        """应该能保存到文件。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_distributions
        import tempfile
        import os

        layer_scores = {
            'layer1': np.random.exponential(1.0, 1000),
        }

        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                save_path = f.name

            fig = plot_layer_distributions(layer_scores, save_path=save_path)
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0

            import matplotlib.pyplot as plt
            plt.close(fig)
            os.unlink(save_path)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_empty_layer_raises_error(self):
        """空层数据应该抛出错误。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_distributions

        layer_scores = {
            'empty_layer': np.array([]),
        }

        try:
            with pytest.raises(ValueError):
                plot_layer_distributions(layer_scores)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_empty_dict_raises_error(self):
        """空字典应该抛出错误。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_distributions

        try:
            with pytest.raises(ValueError):
                plot_layer_distributions({})
        except ImportError:
            pytest.skip("matplotlib not available")


class TestFitAndPlotDistribution:
    """测试 fit_and_plot_distribution 函数。"""

    def test_returns_best_fit_info(self):
        """应该返回最佳拟合信息。"""
        from ckpt_compress.methods.adam_prune.visualization import fit_and_plot_distribution

        data = np.random.exponential(1.0, 1000)

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            result = fit_and_plot_distribution(ax, data, "test_layer")

            assert 'best_dist' in result
            assert 'ks_stat' in result
            assert 'params' in result

            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_fits_exponential_data(self):
        """对指数分布数据应该识别为指数分布或相近分布。"""
        from ckpt_compress.methods.adam_prune.visualization import fit_and_plot_distribution

        np.random.seed(42)
        data = np.random.exponential(1.0, 5000)

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            result = fit_and_plot_distribution(ax, data, "exponential_data")

            # 应该识别为指数分布或韦伯分布（韦伯分布包含指数分布作为特例）
            assert result['best_dist'] in ['exponential', 'weibull', 'gamma']
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_fits_lognormal_data(self):
        """对对数正态分布数据应该识别为对数正态分布。"""
        from ckpt_compress.methods.adam_prune.visualization import fit_and_plot_distribution

        np.random.seed(42)
        data = np.random.lognormal(0, 1, 5000)

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            result = fit_and_plot_distribution(ax, data, "lognormal_data")

            # 应该识别为对数正态分布
            assert result['best_dist'] == 'lognormal'
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_fits_weibull_data(self):
        """对韦伯分布数据应该识别为韦伯分布。"""
        from ckpt_compress.methods.adam_prune.visualization import fit_and_plot_distribution

        np.random.seed(42)
        data = np.random.weibull(1.5, 5000) * 2  # scale=2

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            result = fit_and_plot_distribution(ax, data, "weibull_data")

            # 应该识别为韦伯分布
            assert result['best_dist'] == 'weibull'
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_handles_small_data(self):
        """应该能处理小数据集。"""
        from ckpt_compress.methods.adam_prune.visualization import fit_and_plot_distribution

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            result = fit_and_plot_distribution(ax, data, "small_data")

            assert result is not None
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plots_histogram(self):
        """应该绘制直方图。"""
        from ckpt_compress.methods.adam_prune.visualization import fit_and_plot_distribution

        data = np.random.exponential(1.0, 1000)

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fit_and_plot_distribution(ax, data, "test")

            # 检查是否有绑定的 patches（直方图的条形）
            assert len(ax.patches) > 0
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plots_fitted_curve(self):
        """应该绘制拟合曲线。"""
        from ckpt_compress.methods.adam_prune.visualization import fit_and_plot_distribution

        data = np.random.exponential(1.0, 1000)

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fit_and_plot_distribution(ax, data, "test")

            # 检查是否有线条（拟合曲线）
            assert len(ax.lines) > 0
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestGetDistributionFits:
    """测试 get_distribution_fits 函数。"""

    def test_returns_dict(self):
        """应该返回字典。"""
        from ckpt_compress.methods.adam_prune.visualization import get_distribution_fits

        data = np.random.exponential(1.0, 1000)
        result = get_distribution_fits(data)

        assert isinstance(result, dict)

    def test_contains_multiple_distributions(self):
        """应该包含多种分布的拟合结果。"""
        from ckpt_compress.methods.adam_prune.visualization import get_distribution_fits

        data = np.random.exponential(1.0, 1000)
        result = get_distribution_fits(data)

        expected_dists = ['exponential', 'lognormal', 'weibull', 'gamma']
        for dist in expected_dists:
            assert dist in result

    def test_each_fit_has_required_keys(self):
        """每个拟合结果应该包含必需的键。"""
        from ckpt_compress.methods.adam_prune.visualization import get_distribution_fits

        data = np.random.exponential(1.0, 1000)
        result = get_distribution_fits(data)

        for dist_name, fit_result in result.items():
            if 'error' not in fit_result:
                assert 'ks_stat' in fit_result
                assert 'p_value' in fit_result
                assert 'params' in fit_result

    def test_handles_zeros(self):
        """应该能处理包含零的数据。"""
        from ckpt_compress.methods.adam_prune.visualization import get_distribution_fits

        data = np.array([0, 0, 1, 2, 3, 4, 5])
        result = get_distribution_fits(data)

        assert isinstance(result, dict)

    def test_handles_negative_values(self):
        """应该能处理包含负值的数据（过滤掉）。"""
        from ckpt_compress.methods.adam_prune.visualization import get_distribution_fits

        data = np.array([-1, -2, 1, 2, 3, 4, 5])
        result = get_distribution_fits(data)

        assert isinstance(result, dict)


class TestPlotComparisonGrid:
    """测试 plot_comparison_grid 函数。"""

    def test_returns_figure(self):
        """应该返回 matplotlib figure。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_comparison_grid

        layer_scores = {
            'layer1': np.random.exponential(1.0, 1000),
            'layer2': np.random.lognormal(0, 1, 1000),
        }

        try:
            fig = plot_comparison_grid(layer_scores)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_creates_grid_layout(self):
        """应该创建网格布局。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_comparison_grid

        layer_scores = {
            'layer1': np.random.exponential(1.0, 1000),
            'layer2': np.random.lognormal(0, 1, 1000),
            'layer3': np.random.weibull(1.5, 1000),
            'layer4': np.random.gamma(2, 1, 1000),
        }

        try:
            fig = plot_comparison_grid(layer_scores)
            # 4 个层应该创建 2x2 的网格
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_save_to_file(self):
        """应该能保存到文件。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_comparison_grid
        import tempfile
        import os

        layer_scores = {
            'layer1': np.random.exponential(1.0, 1000),
            'layer2': np.random.lognormal(0, 1, 1000),
        }

        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                save_path = f.name

            fig = plot_comparison_grid(layer_scores, save_path=save_path)
            assert os.path.exists(save_path)

            import matplotlib.pyplot as plt
            plt.close(fig)
            os.unlink(save_path)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_with_figsize(self):
        """应该支持自定义图像大小。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_comparison_grid

        layer_scores = {
            'layer1': np.random.exponential(1.0, 1000),
        }

        try:
            fig = plot_comparison_grid(layer_scores, figsize=(10, 8))
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestVisualizationEdgeCases:
    """测试可视化的边界情况。"""

    def test_very_small_values(self):
        """应该能处理非常小的值。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_distributions

        layer_scores = {
            'small_values': np.random.exponential(1e-10, 1000),
        }

        try:
            fig = plot_layer_distributions(layer_scores)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_very_large_values(self):
        """应该能处理非常大的值。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_distributions

        layer_scores = {
            'large_values': np.random.exponential(1e10, 1000),
        }

        try:
            fig = plot_layer_distributions(layer_scores)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_mixed_scale_layers(self):
        """应该能处理不同尺度的层。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_distributions

        layer_scores = {
            'small': np.random.exponential(1e-8, 1000),
            'medium': np.random.exponential(1.0, 1000),
            'large': np.random.exponential(1e8, 1000),
        }

        try:
            fig = plot_layer_distributions(layer_scores)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_tensor_input(self):
        """应该能处理 torch.Tensor 输入。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_distributions

        layer_scores = {
            'tensor_layer': torch.rand(1000).numpy(),
        }

        try:
            fig = plot_layer_distributions(layer_scores)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestPlotMarginalLossCurves:
    """测试 plot_marginal_loss_curves 函数。"""

    def test_returns_figure(self):
        """应该返回 matplotlib figure。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_marginal_loss_curves

        # 创建测试数据：每个公式的边际损失曲线
        marginal_data = {
            'A': {'prune_ratios': [0.0, 0.1, 0.2, 0.3], 'losses': [0.0, 0.01, 0.03, 0.08]},
            'B': {'prune_ratios': [0.0, 0.1, 0.2, 0.3], 'losses': [0.0, 0.02, 0.05, 0.12]},
            'C': {'prune_ratios': [0.0, 0.1, 0.2, 0.3], 'losses': [0.0, 0.015, 0.04, 0.10]},
        }

        try:
            fig = plot_marginal_loss_curves(marginal_data)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_handles_single_formula(self):
        """应该能处理单个公式。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_marginal_loss_curves

        marginal_data = {
            'A': {'prune_ratios': [0.0, 0.1, 0.2], 'losses': [0.0, 0.01, 0.03]},
        }

        try:
            fig = plot_marginal_loss_curves(marginal_data)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_save_to_file(self):
        """应该能保存到文件。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_marginal_loss_curves
        import tempfile
        import os

        marginal_data = {
            'A': {'prune_ratios': [0.0, 0.1, 0.2], 'losses': [0.0, 0.01, 0.03]},
        }

        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                save_path = f.name

            fig = plot_marginal_loss_curves(marginal_data, save_path=save_path)
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0

            import matplotlib.pyplot as plt
            plt.close(fig)
            os.unlink(save_path)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_empty_data_raises_error(self):
        """空数据应该抛出错误。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_marginal_loss_curves

        try:
            with pytest.raises(ValueError):
                plot_marginal_loss_curves({})
        except ImportError:
            pytest.skip("matplotlib not available")


class TestPlotPredictionVsActual:
    """测试 plot_prediction_vs_actual 函数。"""

    def test_returns_figure(self):
        """应该返回 matplotlib figure。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_prediction_vs_actual

        # 创建测试数据
        results = {
            'A': {'predicted': [0.01, 0.02, 0.05], 'actual': [0.012, 0.022, 0.048]},
            'B': {'predicted': [0.01, 0.02, 0.05], 'actual': [0.015, 0.025, 0.055]},
        }

        try:
            fig = plot_prediction_vs_actual(results)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plots_diagonal_line(self):
        """应该绘制对角线（理想预测线）。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_prediction_vs_actual

        results = {
            'A': {'predicted': [0.01, 0.02], 'actual': [0.01, 0.02]},
        }

        try:
            import matplotlib.pyplot as plt
            fig = plot_prediction_vs_actual(results)
            ax = fig.axes[0]
            # 检查是否有线条（对角线）
            assert len(ax.lines) > 0
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_save_to_file(self):
        """应该能保存到文件。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_prediction_vs_actual
        import tempfile
        import os

        results = {
            'A': {'predicted': [0.01, 0.02], 'actual': [0.012, 0.022]},
        }

        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                save_path = f.name

            fig = plot_prediction_vs_actual(results, save_path=save_path)
            assert os.path.exists(save_path)

            import matplotlib.pyplot as plt
            plt.close(fig)
            os.unlink(save_path)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_empty_data_raises_error(self):
        """空数据应该抛出错误。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_prediction_vs_actual

        try:
            with pytest.raises(ValueError):
                plot_prediction_vs_actual({})
        except ImportError:
            pytest.skip("matplotlib not available")


class TestPlotLayerAllocationHeatmap:
    """测试 plot_layer_allocation_heatmap 函数。"""

    def test_returns_figure(self):
        """应该返回 matplotlib figure。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_allocation_heatmap

        # 创建测试数据：每个公式每层的剪枝比例
        allocation_data = {
            'A': {'layer1': 0.1, 'layer2': 0.2, 'layer3': 0.15},
            'B': {'layer1': 0.12, 'layer2': 0.18, 'layer3': 0.2},
            'C': {'layer1': 0.11, 'layer2': 0.19, 'layer3': 0.17},
        }

        try:
            fig = plot_layer_allocation_heatmap(allocation_data)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_handles_many_layers(self):
        """应该能处理多层。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_allocation_heatmap

        allocation_data = {
            'A': {f'layer{i}': 0.1 + i * 0.01 for i in range(10)},
        }

        try:
            fig = plot_layer_allocation_heatmap(allocation_data)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_save_to_file(self):
        """应该能保存到文件。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_allocation_heatmap
        import tempfile
        import os

        allocation_data = {
            'A': {'layer1': 0.1, 'layer2': 0.2},
        }

        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                save_path = f.name

            fig = plot_layer_allocation_heatmap(allocation_data, save_path=save_path)
            assert os.path.exists(save_path)

            import matplotlib.pyplot as plt
            plt.close(fig)
            os.unlink(save_path)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_empty_data_raises_error(self):
        """空数据应该抛出错误。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_layer_allocation_heatmap

        try:
            with pytest.raises(ValueError):
                plot_layer_allocation_heatmap({})
        except ImportError:
            pytest.skip("matplotlib not available")


class TestPlotFormulaComparison:
    """测试 plot_formula_comparison 函数。"""

    def test_returns_figure(self):
        """应该返回 matplotlib figure。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_formula_comparison

        # 创建测试数据：每个公式的误差指标
        comparison_data = {
            'A': {'mean_relative_error': 0.15, 'max_relative_error': 0.25, 'global_sparsity': 0.2},
            'B': {'mean_relative_error': 0.20, 'max_relative_error': 0.35, 'global_sparsity': 0.22},
            'C': {'mean_relative_error': 0.18, 'max_relative_error': 0.30, 'global_sparsity': 0.21},
        }

        try:
            fig = plot_formula_comparison(comparison_data)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_creates_bar_chart(self):
        """应该创建柱状图。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_formula_comparison

        comparison_data = {
            'A': {'mean_relative_error': 0.15},
            'B': {'mean_relative_error': 0.20},
        }

        try:
            import matplotlib.pyplot as plt
            fig = plot_formula_comparison(comparison_data)
            ax = fig.axes[0]
            # 检查是否有柱状图的 patches
            assert len(ax.patches) > 0
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_save_to_file(self):
        """应该能保存到文件。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_formula_comparison
        import tempfile
        import os

        comparison_data = {
            'A': {'mean_relative_error': 0.15},
        }

        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                save_path = f.name

            fig = plot_formula_comparison(comparison_data, save_path=save_path)
            assert os.path.exists(save_path)

            import matplotlib.pyplot as plt
            plt.close(fig)
            os.unlink(save_path)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_empty_data_raises_error(self):
        """空数据应该抛出错误。"""
        from ckpt_compress.methods.adam_prune.visualization import plot_formula_comparison

        try:
            with pytest.raises(ValueError):
                plot_formula_comparison({})
        except ImportError:
            pytest.skip("matplotlib not available")
