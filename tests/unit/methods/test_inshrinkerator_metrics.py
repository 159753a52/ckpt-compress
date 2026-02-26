"""
Inshrinkerator 指标模块测试（幅度和敏感度）。
"""

import pytest
import torch

from ckpt_compress.methods.inshrinkerator.metrics import magnitude, sensitivity


class TestMagnitude:
    """幅度计算测试。"""

    def test_magnitude_basic(self):
        """基本幅度测试：I_m(w) = |w|。"""
        w = torch.tensor([-2.0, 0.0, 3.0])

        result = magnitude(w)

        expected = torch.tensor([2.0, 0.0, 3.0])
        assert torch.allclose(result, expected)

    def test_magnitude_preserves_shape(self):
        """幅度应保持张量形状。"""
        w = torch.randn(64, 32)

        result = magnitude(w)

        assert result.shape == w.shape

    def test_magnitude_all_positive(self):
        """幅度应始终为非负。"""
        w = torch.randn(100)

        result = magnitude(w)

        assert torch.all(result >= 0)

    def test_magnitude_zero_tensor(self):
        """零张量的幅度应为零。"""
        w = torch.zeros(10)

        result = magnitude(w)

        assert torch.all(result == 0)


class TestSensitivity:
    """敏感度计算测试。"""

    def test_sensitivity_basic(self):
        """基本敏感度测试：I_s(w) = |grad * w|。"""
        w = torch.tensor([2.0, -3.0])
        grad = torch.tensor([0.1, 0.2])

        result = sensitivity(w, grad)

        # |0.1 * 2| = 0.2, |0.2 * -3| = 0.6
        expected = torch.tensor([0.2, 0.6])
        assert torch.allclose(result, expected)

    def test_sensitivity_preserves_shape(self):
        """敏感度应保持张量形状。"""
        w = torch.randn(64, 32)
        grad = torch.randn(64, 32)

        result = sensitivity(w, grad)

        assert result.shape == w.shape

    def test_sensitivity_all_positive(self):
        """敏感度应始终为非负。"""
        w = torch.randn(100)
        grad = torch.randn(100)

        result = sensitivity(w, grad)

        assert torch.all(result >= 0)

    def test_sensitivity_zero_grad(self):
        """零梯度的敏感度应为零。"""
        w = torch.randn(10)
        grad = torch.zeros(10)

        result = sensitivity(w, grad)

        assert torch.all(result == 0)

    def test_sensitivity_zero_weight(self):
        """零权重的敏感度应为零。"""
        w = torch.zeros(10)
        grad = torch.randn(10)

        result = sensitivity(w, grad)

        assert torch.all(result == 0)


class TestMetricsCombined:
    """组合指标使用测试。"""

    def test_high_magnitude_low_sensitivity(self):
        """测试用例：高幅度但低敏感度。"""
        w = torch.tensor([10.0, 0.1])  # 第一个是高幅度
        grad = torch.tensor([0.001, 1.0])  # 第一个梯度低

        mag = magnitude(w)
        sens = sensitivity(w, grad)

        # 第一个元素：高幅度 (10)，低敏感度 (0.01)
        # 第二个元素：低幅度 (0.1)，高敏感度 (0.1)
        assert mag[0] > mag[1]  # 第一个幅度更高
        assert sens[0] < sens[1]  # 第一个敏感度更低

    def test_low_magnitude_high_sensitivity(self):
        """测试用例：低幅度但高敏感度。"""
        w = torch.tensor([0.1, 10.0])
        grad = torch.tensor([10.0, 0.001])

        mag = magnitude(w)
        sens = sensitivity(w, grad)

        # 第一个元素：低幅度 (0.1)，高敏感度 (1.0)
        # 第二个元素：高幅度 (10)，低敏感度 (0.01)
        assert mag[0] < mag[1]
        assert sens[0] > sens[1]

    def test_metrics_dtype_consistency(self):
        """指标应保持 dtype。"""
        w = torch.randn(10, dtype=torch.float32)
        grad = torch.randn(10, dtype=torch.float32)

        mag = magnitude(w)
        sens = sensitivity(w, grad)

        assert mag.dtype == torch.float32
        assert sens.dtype == torch.float32
