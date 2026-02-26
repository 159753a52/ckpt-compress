"""
张量操作测试: flatten/unflatten 状态字典工具函数。
"""

import pytest
import torch
from typing import Dict

from ckpt_compress.utils.tensor_ops import flatten_state_dict, unflatten_state_dict


class TestFlattenUnflatten:
    """状态字典 flatten/unflatten 操作的测试。"""

    def test_flatten_unflatten_identity_fp32(self):
        """Flatten->unflatten 应该对 fp32 返回相同的张量。"""
        state_dict = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
        }
        flat_vector, spec = flatten_state_dict(state_dict)
        restored = unflatten_state_dict(flat_vector, spec)

        for key in state_dict:
            assert torch.allclose(state_dict[key], restored[key])

    def test_flatten_preserves_ordering(self):
        """Flatten 应该保持键的顺序并可预测地拼接。"""
        state_dict = {
            "a": torch.tensor([1.0, 2.0]),
            "b": torch.tensor([3.0, 4.0, 5.0]),
        }
        flat_vector, spec = flatten_state_dict(state_dict)

        # 应该按顺序为 [1, 2, 3, 4, 5]
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        assert torch.allclose(flat_vector, expected)

    def test_unflatten_shape_mismatch_raises(self):
        """当向量长度与 spec 不匹配时，unflatten 应该抛出错误。"""
        state_dict = {"a": torch.randn(10)}
        _, spec = flatten_state_dict(state_dict)

        wrong_vector = torch.randn(5)  # 错误的长度
        with pytest.raises(ValueError):
            unflatten_state_dict(wrong_vector, spec)

    def test_supports_empty_tensors(self):
        """应该处理零元素的张量。"""
        state_dict = {
            "empty": torch.tensor([]),
            "normal": torch.randn(5),
        }
        flat_vector, spec = flatten_state_dict(state_dict)
        restored = unflatten_state_dict(flat_vector, spec)

        assert restored["empty"].shape == torch.Size([0])
        assert torch.allclose(state_dict["normal"], restored["normal"])

    def test_dtype_roundtrip(self):
        """应该在 flatten/unflatten 过程中保持 dtype 信息。"""
        state_dict = {
            "fp32": torch.randn(10, dtype=torch.float32),
            "fp16": torch.randn(10, dtype=torch.float16),
        }
        flat_vector, spec = flatten_state_dict(state_dict)
        restored = unflatten_state_dict(flat_vector, spec)

        assert restored["fp32"].dtype == torch.float32
        assert restored["fp16"].dtype == torch.float16

    def test_sparse_keys_ignored_or_handled(self):
        """非张量项应该被处理（单独存储或忽略）。"""
        state_dict = {
            "weight": torch.randn(10),
            "step": 100,  # 非张量
            "rng_state": None,  # 非张量
        }
        flat_vector, spec = flatten_state_dict(state_dict)
        restored = unflatten_state_dict(flat_vector, spec)

        # 张量应该被恢复
        assert torch.allclose(state_dict["weight"], restored["weight"])
        # 非张量应该在 spec 中保留并恢复
        assert restored.get("step") == 100 or "step" not in restored
        # 实现可以选择忽略或保留非张量


class TestFlattenEdgeCases:
    """flatten/unflatten 的边界情况测试。"""

    def test_empty_state_dict(self):
        """应该处理空状态字典。"""
        state_dict = {}
        flat_vector, spec = flatten_state_dict(state_dict)
        restored = unflatten_state_dict(flat_vector, spec)

        assert len(restored) == 0
        assert flat_vector.numel() == 0

    def test_single_scalar_tensor(self):
        """应该处理标量张量。"""
        state_dict = {"scalar": torch.tensor(3.14)}
        flat_vector, spec = flatten_state_dict(state_dict)
        restored = unflatten_state_dict(flat_vector, spec)

        assert torch.allclose(state_dict["scalar"], restored["scalar"])

    def test_various_shapes(self):
        """应该处理各种形状的张量。"""
        state_dict = {
            "1d": torch.randn(100),
            "2d": torch.randn(10, 20),
            "3d": torch.randn(2, 3, 4),
            "4d": torch.randn(2, 3, 4, 5),
        }
        flat_vector, spec = flatten_state_dict(state_dict)
        restored = unflatten_state_dict(flat_vector, spec)

        for key in state_dict:
            assert state_dict[key].shape == restored[key].shape
            assert torch.allclose(state_dict[key], restored[key])

    def test_large_state_dict(self):
        """应该高效处理较大的状态字典。"""
        state_dict = {
            f"layer{i}.weight": torch.randn(256, 256)
            for i in range(10)
        }
        flat_vector, spec = flatten_state_dict(state_dict)
        restored = unflatten_state_dict(flat_vector, spec)

        for key in state_dict:
            assert torch.allclose(state_dict[key], restored[key])
