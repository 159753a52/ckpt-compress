"""
预测残差压缩 - 自适应量化模块测试。

TDD：先写测试，再实现。

关键特性：
- 敏感度感知量化（重要参数分配更多比特）
- 混合精度量化
- 高效编码
"""

import pytest
import torch

from ckpt_compress.methods.predictive.adaptive_quantization import (
    compute_sensitivity,
    assign_bit_allocation,
    adaptive_quantize,
    adaptive_dequantize,
    AdaptiveQuantizer,
)


class TestComputeSensitivity:
    """敏感度计算测试。"""

    def test_sensitivity_from_gradient(self):
        """敏感度 = |grad| * |weight|。"""
        weight = torch.tensor([1.0, 2.0, 3.0])
        grad = torch.tensor([0.1, 0.2, 0.3])

        sensitivity = compute_sensitivity(weight, grad)

        # sensitivity = |grad| * |weight|
        expected = torch.tensor([0.1, 0.4, 0.9])
        assert torch.allclose(sensitivity, expected, atol=1e-6)

    def test_sensitivity_zero_grad(self):
        """零梯度意味着零敏感度。"""
        weight = torch.tensor([1.0, 2.0, 3.0])
        grad = torch.zeros(3)

        sensitivity = compute_sensitivity(weight, grad)

        assert torch.allclose(sensitivity, torch.zeros(3))

    def test_sensitivity_negative_values(self):
        """应正确处理负值。"""
        weight = torch.tensor([-1.0, 2.0, -3.0])
        grad = torch.tensor([0.1, -0.2, 0.3])

        sensitivity = compute_sensitivity(weight, grad)

        # 使用绝对值
        expected = torch.tensor([0.1, 0.4, 0.9])
        assert torch.allclose(sensitivity, expected, atol=1e-6)


class TestAssignBitAllocation:
    """基于敏感度的比特分配测试。"""

    def test_bit_allocation_basic(self):
        """高敏感度应分配更多比特。"""
        sensitivity = torch.tensor([0.1, 0.5, 0.9, 0.2])
        min_bits = 2
        max_bits = 8

        bits = assign_bit_allocation(sensitivity, min_bits, max_bits)

        # 高敏感度应分配更多比特
        assert bits[2] >= bits[0]  # 0.9 > 0.1
        assert bits[1] >= bits[3]  # 0.5 > 0.2
        assert torch.all(bits >= min_bits)
        assert torch.all(bits <= max_bits)

    def test_bit_allocation_uniform_sensitivity(self):
        """均匀敏感度应给出均匀比特。"""
        sensitivity = torch.ones(10) * 0.5
        min_bits = 4
        max_bits = 8

        bits = assign_bit_allocation(sensitivity, min_bits, max_bits)

        # 所有应相同
        assert torch.all(bits == bits[0])

    def test_bit_allocation_respects_bounds(self):
        """比特应在 [min_bits, max_bits] 范围内。"""
        sensitivity = torch.tensor([0.0, 0.5, 1.0, 100.0])
        min_bits = 2
        max_bits = 8

        bits = assign_bit_allocation(sensitivity, min_bits, max_bits)

        assert torch.all(bits >= min_bits)
        assert torch.all(bits <= max_bits)


class TestAdaptiveQuantize:
    """自适应量化测试。"""

    def test_quantize_basic(self):
        """量化应产生整数索引。"""
        values = torch.tensor([0.1, 0.5, 0.9, -0.3])
        n_bits = 4

        indices, scale, zero_point = adaptive_quantize(values, n_bits)

        assert indices.dtype == torch.int32 or indices.dtype == torch.long
        assert torch.all(indices >= 0)
        assert torch.all(indices < 2**n_bits)

    def test_quantize_dequantize_roundtrip(self):
        """量化后反量化应近似原始值。"""
        torch.manual_seed(42)
        values = torch.randn(100)
        n_bits = 8

        indices, scale, zero_point = adaptive_quantize(values, n_bits)
        reconstructed = adaptive_dequantize(indices, scale, zero_point)

        # 应接近原始值
        mse = torch.mean((values - reconstructed) ** 2)
        assert mse < 0.01  # 8 比特的合理误差

    def test_quantize_higher_bits_lower_error(self):
        """更多比特应产生更低的量化误差。"""
        torch.manual_seed(42)
        values = torch.randn(100)

        # 4 比特量化
        indices_4, scale_4, zp_4 = adaptive_quantize(values, 4)
        recon_4 = adaptive_dequantize(indices_4, scale_4, zp_4)
        mse_4 = torch.mean((values - recon_4) ** 2)

        # 8 比特量化
        indices_8, scale_8, zp_8 = adaptive_quantize(values, 8)
        recon_8 = adaptive_dequantize(indices_8, scale_8, zp_8)
        mse_8 = torch.mean((values - recon_8) ** 2)

        assert mse_8 < mse_4  # 8 比特应更精确


class TestAdaptiveQuantizer:
    """AdaptiveQuantizer 类测试。"""

    def test_quantizer_init(self):
        """量化器应使用配置初始化。"""
        quantizer = AdaptiveQuantizer(min_bits=2, max_bits=8)

        assert quantizer.min_bits == 2
        assert quantizer.max_bits == 8

    def test_quantizer_quantize_tensor(self):
        """量化器应量化单个张量。"""
        torch.manual_seed(42)
        quantizer = AdaptiveQuantizer(min_bits=4, max_bits=8)

        tensor = torch.randn(32, 32)
        grad = torch.randn(32, 32)

        result = quantizer.quantize(tensor, grad)

        assert "indices" in result
        assert "scale" in result
        assert "zero_point" in result
        assert "bits" in result

    def test_quantizer_dequantize_tensor(self):
        """量化器应反量化回张量。"""
        torch.manual_seed(42)
        quantizer = AdaptiveQuantizer(min_bits=4, max_bits=8)

        tensor = torch.randn(32, 32)
        grad = torch.randn(32, 32)

        result = quantizer.quantize(tensor, grad)
        reconstructed = quantizer.dequantize(result)

        assert reconstructed.shape == tensor.shape

    def test_quantizer_roundtrip_preserves_shape(self):
        """量化-反量化应保持形状。"""
        torch.manual_seed(42)
        quantizer = AdaptiveQuantizer()

        tensor = torch.randn(64, 128)
        grad = torch.randn(64, 128)

        result = quantizer.quantize(tensor, grad)
        reconstructed = quantizer.dequantize(result)

        assert reconstructed.shape == tensor.shape

    def test_quantizer_state_dict(self):
        """量化器应处理状态字典。"""
        torch.manual_seed(42)
        quantizer = AdaptiveQuantizer(min_bits=4, max_bits=8)

        state_dict = {
            "layer1.weight": torch.randn(32, 32),
            "layer1.bias": torch.randn(32),
        }
        grad_dict = {
            "layer1.weight": torch.randn(32, 32),
            "layer1.bias": torch.randn(32),
        }

        compressed = quantizer.quantize_state_dict(state_dict, grad_dict)

        assert "layer1.weight" in compressed
        assert "layer1.bias" in compressed

    def test_quantizer_state_dict_roundtrip(self):
        """状态字典量化-反量化应保持结构。"""
        torch.manual_seed(42)
        quantizer = AdaptiveQuantizer(min_bits=4, max_bits=8)

        state_dict = {
            "layer1.weight": torch.randn(32, 32),
            "layer1.bias": torch.randn(32),
        }
        grad_dict = {
            "layer1.weight": torch.randn(32, 32),
            "layer1.bias": torch.randn(32),
        }

        compressed = quantizer.quantize_state_dict(state_dict, grad_dict)
        reconstructed = quantizer.dequantize_state_dict(compressed)

        assert set(reconstructed.keys()) == set(state_dict.keys())
        for key in state_dict:
            assert reconstructed[key].shape == state_dict[key].shape

    def test_quantizer_high_sensitivity_preserved_better(self):
        """高敏感度值应有更低的误差。"""
        torch.manual_seed(42)
        quantizer = AdaptiveQuantizer(min_bits=2, max_bits=8)

        # 创建具有已知结构的张量
        tensor = torch.randn(100)
        # 前半部分高梯度，后半部分低梯度
        grad = torch.cat([torch.ones(50) * 10, torch.ones(50) * 0.01])

        result = quantizer.quantize(tensor, grad)
        reconstructed = quantizer.dequantize(result)

        # 高敏感度区域的误差应更低
        error_high = torch.mean((tensor[:50] - reconstructed[:50]) ** 2)
        error_low = torch.mean((tensor[50:] - reconstructed[50:]) ** 2)

        # 高敏感度区域应有更低的误差（分配更多比特）
        assert error_high < error_low
