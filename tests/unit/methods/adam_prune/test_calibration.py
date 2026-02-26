"""
校正模块测试。

测试幂律拟合、逆映射和评估指标计算。
"""

import pytest
import numpy as np
from typing import List


class TestFitPowerlaw:
    """测试幂律拟合函数。"""

    def test_fit_powerlaw_basic(self):
        """测试基本的幂律拟合。"""
        from ckpt_compress.methods.adam_prune.calibration import fit_powerlaw

        # 生成 y = 2 * x^0.5 的数据
        xs = [0.01, 0.04, 0.09, 0.16, 0.25]
        ys = [2 * (x ** 0.5) for x in xs]  # [0.2, 0.4, 0.6, 0.8, 1.0]

        a, b = fit_powerlaw(xs, ys)

        # 检查拟合参数
        assert abs(a - 2.0) < 0.1, f"Expected a ≈ 2.0, got {a}"
        assert abs(b - 0.5) < 0.1, f"Expected b ≈ 0.5, got {b}"

    def test_fit_powerlaw_linear(self):
        """测试线性关系 (b=1)。"""
        from ckpt_compress.methods.adam_prune.calibration import fit_powerlaw

        # y = 3 * x^1
        xs = [0.1, 0.2, 0.3, 0.4, 0.5]
        ys = [3 * x for x in xs]

        a, b = fit_powerlaw(xs, ys)

        assert abs(a - 3.0) < 0.1, f"Expected a ≈ 3.0, got {a}"
        assert abs(b - 1.0) < 0.1, f"Expected b ≈ 1.0, got {b}"

    def test_fit_powerlaw_quadratic(self):
        """测试二次关系 (b=2)。"""
        from ckpt_compress.methods.adam_prune.calibration import fit_powerlaw

        # y = 1.5 * x^2
        xs = [0.1, 0.2, 0.3, 0.4, 0.5]
        ys = [1.5 * (x ** 2) for x in xs]

        a, b = fit_powerlaw(xs, ys)

        assert abs(a - 1.5) < 0.2, f"Expected a ≈ 1.5, got {a}"
        assert abs(b - 2.0) < 0.2, f"Expected b ≈ 2.0, got {b}"

    def test_fit_powerlaw_with_noise(self):
        """测试带噪声数据的拟合。"""
        from ckpt_compress.methods.adam_prune.calibration import fit_powerlaw

        np.random.seed(42)
        # y = 2 * x^0.8 + noise
        xs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        ys = [2 * (x ** 0.8) + np.random.normal(0, 0.01) for x in xs]

        a, b = fit_powerlaw(xs, ys)

        # 允许更大的误差范围
        assert abs(a - 2.0) < 0.5, f"Expected a ≈ 2.0, got {a}"
        assert abs(b - 0.8) < 0.3, f"Expected b ≈ 0.8, got {b}"

    def test_fit_powerlaw_empty_input(self):
        """测试空输入。"""
        from ckpt_compress.methods.adam_prune.calibration import fit_powerlaw

        with pytest.raises(ValueError):
            fit_powerlaw([], [])

    def test_fit_powerlaw_single_point(self):
        """测试单点输入。"""
        from ckpt_compress.methods.adam_prune.calibration import fit_powerlaw

        with pytest.raises(ValueError):
            fit_powerlaw([0.1], [0.2])

    def test_fit_powerlaw_mismatched_lengths(self):
        """测试长度不匹配的输入。"""
        from ckpt_compress.methods.adam_prune.calibration import fit_powerlaw

        with pytest.raises(ValueError):
            fit_powerlaw([0.1, 0.2], [0.1])

    def test_fit_powerlaw_negative_values(self):
        """测试负值输入（应该过滤或报错）。"""
        from ckpt_compress.methods.adam_prune.calibration import fit_powerlaw

        # 负值在对数空间无法处理
        with pytest.raises(ValueError):
            fit_powerlaw([-0.1, 0.2], [0.1, 0.2])

    def test_fit_powerlaw_zero_values(self):
        """测试零值输入（应该过滤或报错）。"""
        from ckpt_compress.methods.adam_prune.calibration import fit_powerlaw

        # 零值在对数空间无法处理
        with pytest.raises(ValueError):
            fit_powerlaw([0.0, 0.2], [0.1, 0.2])


class TestInvertPowerlaw:
    """测试幂律逆映射函数。"""

    def test_invert_powerlaw_basic(self):
        """测试基本的逆映射。"""
        from ckpt_compress.methods.adam_prune.calibration import invert_powerlaw

        # y = 2 * x^0.5 => x = (y/2)^2
        a, b = 2.0, 0.5
        y_target = 1.0

        x = invert_powerlaw(y_target, a, b)
        expected_x = (y_target / a) ** (1 / b)  # (1/2)^2 = 0.25

        assert abs(x - expected_x) < 1e-6, f"Expected {expected_x}, got {x}"

    def test_invert_powerlaw_linear(self):
        """测试线性逆映射 (b=1)。"""
        from ckpt_compress.methods.adam_prune.calibration import invert_powerlaw

        # y = 3 * x => x = y/3
        a, b = 3.0, 1.0
        y_target = 0.9

        x = invert_powerlaw(y_target, a, b)
        expected_x = 0.3

        assert abs(x - expected_x) < 1e-6, f"Expected {expected_x}, got {x}"

    def test_invert_powerlaw_quadratic(self):
        """测试二次逆映射 (b=2)。"""
        from ckpt_compress.methods.adam_prune.calibration import invert_powerlaw

        # y = 1.5 * x^2 => x = sqrt(y/1.5)
        a, b = 1.5, 2.0
        y_target = 0.6

        x = invert_powerlaw(y_target, a, b)
        expected_x = np.sqrt(y_target / a)

        assert abs(x - expected_x) < 1e-6, f"Expected {expected_x}, got {x}"

    def test_invert_powerlaw_roundtrip(self):
        """测试正向和逆向映射的一致性。"""
        from ckpt_compress.methods.adam_prune.calibration import invert_powerlaw

        a, b = 2.5, 0.7
        x_original = 0.15

        # 正向: y = a * x^b
        y = a * (x_original ** b)

        # 逆向: x = (y/a)^(1/b)
        x_recovered = invert_powerlaw(y, a, b)

        assert abs(x_recovered - x_original) < 1e-6

    def test_invert_powerlaw_zero_target(self):
        """测试零目标值。"""
        from ckpt_compress.methods.adam_prune.calibration import invert_powerlaw

        x = invert_powerlaw(0.0, 2.0, 0.5)
        assert x == 0.0

    def test_invert_powerlaw_invalid_a(self):
        """测试无效的 a 参数。"""
        from ckpt_compress.methods.adam_prune.calibration import invert_powerlaw

        with pytest.raises(ValueError):
            invert_powerlaw(1.0, 0.0, 0.5)  # a = 0

        with pytest.raises(ValueError):
            invert_powerlaw(1.0, -1.0, 0.5)  # a < 0

    def test_invert_powerlaw_invalid_b(self):
        """测试无效的 b 参数。"""
        from ckpt_compress.methods.adam_prune.calibration import invert_powerlaw

        with pytest.raises(ValueError):
            invert_powerlaw(1.0, 2.0, 0.0)  # b = 0


class TestComputeCalibrationMetrics:
    """测试校正指标计算函数。"""

    def test_compute_metrics_perfect_fit(self):
        """测试完美拟合的指标。"""
        from ckpt_compress.methods.adam_prune.calibration import compute_calibration_metrics

        # 完美拟合: y = 2 * x^0.5
        a, b = 2.0, 0.5
        xs = [0.01, 0.04, 0.09, 0.16, 0.25]
        ys = [a * (x ** b) for x in xs]

        metrics = compute_calibration_metrics(xs, ys, a, b)

        assert metrics['r_squared'] > 0.99, f"Expected R² > 0.99, got {metrics['r_squared']}"
        assert metrics['rmse'] < 0.01, f"Expected RMSE < 0.01, got {metrics['rmse']}"
        assert metrics['mae'] < 0.01, f"Expected MAE < 0.01, got {metrics['mae']}"
        assert metrics['spearman'] > 0.99, f"Expected Spearman > 0.99, got {metrics['spearman']}"

    def test_compute_metrics_with_noise(self):
        """测试带噪声数据的指标。"""
        from ckpt_compress.methods.adam_prune.calibration import compute_calibration_metrics

        np.random.seed(42)
        a, b = 2.0, 0.5
        xs = [0.01, 0.04, 0.09, 0.16, 0.25]
        ys = [a * (x ** b) + np.random.normal(0, 0.05) for x in xs]

        metrics = compute_calibration_metrics(xs, ys, a, b)

        # 有噪声时指标会下降
        assert 'r_squared' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'spearman' in metrics

    def test_compute_metrics_poor_fit(self):
        """测试差拟合的指标。"""
        from ckpt_compress.methods.adam_prune.calibration import compute_calibration_metrics

        # 使用错误的参数
        a, b = 2.0, 0.5
        xs = [0.01, 0.04, 0.09, 0.16, 0.25]
        # 实际数据是线性的，但用幂律拟合
        ys = [x * 10 for x in xs]

        metrics = compute_calibration_metrics(xs, ys, a, b)

        # R² 应该较低
        assert metrics['r_squared'] < 0.9

    def test_compute_metrics_empty_input(self):
        """测试空输入。"""
        from ckpt_compress.methods.adam_prune.calibration import compute_calibration_metrics

        with pytest.raises(ValueError):
            compute_calibration_metrics([], [], 2.0, 0.5)


class TestComputeCalibrationVariables:
    """测试校正变量计算函数。"""

    def test_compute_calibration_variables_basic(self):
        """测试基本的校正变量计算。"""
        from ckpt_compress.methods.adam_prune.calibration import compute_calibration_variables

        sum_pruned_scores = 10.0
        baseline_loss = 5.0
        actual_loss_increase = 2.5

        result = compute_calibration_variables(
            sum_pruned_scores, baseline_loss, actual_loss_increase
        )

        assert 'x' in result
        assert 'y' in result
        assert abs(result['x'] - 2.0) < 1e-6  # 10 / 5 = 2
        assert abs(result['y'] - 0.5) < 1e-6  # 2.5 / 5 = 0.5

    def test_compute_calibration_variables_zero_baseline(self):
        """测试零基线损失（应该使用小常数避免除零）。"""
        from ckpt_compress.methods.adam_prune.calibration import compute_calibration_variables

        result = compute_calibration_variables(
            sum_pruned_scores=1.0,
            baseline_loss=0.0,
            actual_loss_increase=0.5
        )

        # 应该返回有限值（使用 eps 避免除零）
        assert np.isfinite(result['x'])
        assert np.isfinite(result['y'])

    def test_compute_calibration_variables_negative_loss(self):
        """测试负损失增量（剪枝后损失降低的情况）。"""
        from ckpt_compress.methods.adam_prune.calibration import compute_calibration_variables

        result = compute_calibration_variables(
            sum_pruned_scores=1.0,
            baseline_loss=5.0,
            actual_loss_increase=-0.1  # 损失降低
        )

        assert result['y'] < 0  # y 可以为负


class TestIntegration:
    """集成测试。"""

    def test_fit_and_invert_roundtrip(self):
        """测试拟合和逆映射的完整流程。"""
        from ckpt_compress.methods.adam_prune.calibration import (
            fit_powerlaw, invert_powerlaw, compute_calibration_metrics
        )

        # 模拟实验数据
        np.random.seed(42)
        true_a, true_b = 1.5, 0.6
        xs = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
        ys = [true_a * (x ** true_b) + np.random.normal(0, 0.005) for x in xs]

        # 拟合
        a, b = fit_powerlaw(xs, ys)

        # 验证拟合质量
        metrics = compute_calibration_metrics(xs, ys, a, b)
        assert metrics['r_squared'] > 0.95

        # 使用逆映射预测
        y_target = 0.3
        x_predicted = invert_powerlaw(y_target, a, b)

        # 验证预测的 x 产生接近目标的 y
        y_actual = a * (x_predicted ** b)
        assert abs(y_actual - y_target) < 0.05

    def test_realistic_pruning_scenario(self):
        """测试真实剪枝场景的数据。"""
        from ckpt_compress.methods.adam_prune.calibration import (
            fit_powerlaw, invert_powerlaw, compute_calibration_variables
        )

        # 模拟真实剪枝实验数据
        # 稀疏度: [0.5%, 1%, 2%, 4%, 6%, 8%]
        # 假设 baseline_loss = 3.5
        baseline_loss = 3.5

        # 模拟数据: 随着稀疏度增加，p 和 Δloss 都增加
        experiment_data = [
            {'sparsity': 0.005, 'p': 0.5, 'delta_loss': 0.1},
            {'sparsity': 0.01, 'p': 1.2, 'delta_loss': 0.25},
            {'sparsity': 0.02, 'p': 2.8, 'delta_loss': 0.6},
            {'sparsity': 0.04, 'p': 6.5, 'delta_loss': 1.5},
            {'sparsity': 0.06, 'p': 10.0, 'delta_loss': 2.5},
            {'sparsity': 0.08, 'p': 14.0, 'delta_loss': 3.8},
        ]

        # 计算 x 和 y
        xs = []
        ys = []
        for data in experiment_data:
            vars = compute_calibration_variables(
                data['p'], baseline_loss, data['delta_loss']
            )
            xs.append(vars['x'])
            ys.append(vars['y'])

        # 拟合幂律
        a, b = fit_powerlaw(xs, ys)

        # 验证参数合理
        assert a > 0
        assert b > 0

        # 使用逆映射：给定目标相对损失 y* = 0.5，预测需要的 x
        y_target = 0.5
        x_predicted = invert_powerlaw(y_target, a, b)

        # x 应该在合理范围内
        assert x_predicted > 0
        assert x_predicted < max(xs) * 2  # 不应该外推太远
