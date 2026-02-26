"""
ckpt-compress 测试的 Pytest 配置和共享 fixtures。
"""

import pytest
import torch
from typing import Dict


@pytest.fixture
def simple_state_dict() -> Dict[str, torch.Tensor]:
    """创建用于测试的简单状态字典。"""
    return {
        "layer1.weight": torch.randn(64, 32),
        "layer1.bias": torch.randn(64),
        "layer2.weight": torch.randn(128, 64),
        "layer2.bias": torch.randn(128),
    }


@pytest.fixture
def empty_state_dict() -> Dict[str, torch.Tensor]:
    """创建用于测试的空状态字典。"""
    return {}


@pytest.fixture
def single_tensor_state_dict() -> Dict[str, torch.Tensor]:
    """创建包含单个张量的状态字典。"""
    return {"weight": torch.randn(100, 100)}


@pytest.fixture
def large_state_dict() -> Dict[str, torch.Tensor]:
    """创建用于性能测试的较大状态字典。"""
    return {
        "layer1.weight": torch.randn(512, 256),
        "layer1.bias": torch.randn(512),
        "layer2.weight": torch.randn(1024, 512),
        "layer2.bias": torch.randn(1024),
        "layer3.weight": torch.randn(256, 1024),
        "layer3.bias": torch.randn(256),
    }


@pytest.fixture
def state_dict_with_special_values() -> Dict[str, torch.Tensor]:
    """创建包含特殊值（零、非常小、非常大）的状态字典。"""
    return {
        "zeros": torch.zeros(32, 32),
        "ones": torch.ones(32, 32),
        "small_values": torch.randn(32, 32) * 1e-7,
        "large_values": torch.randn(32, 32) * 1e7,
    }


@pytest.fixture
def state_dict_different_dtypes() -> Dict[str, torch.Tensor]:
    """创建包含不同数据类型的状态字典。"""
    return {
        "float32": torch.randn(32, 32, dtype=torch.float32),
        "float16": torch.randn(32, 32, dtype=torch.float16),
        "bfloat16": torch.randn(32, 32, dtype=torch.bfloat16),
    }


@pytest.fixture
def state_dict_various_shapes() -> Dict[str, torch.Tensor]:
    """创建包含各种张量形状的状态字典。"""
    return {
        "scalar": torch.tensor(1.0),
        "vector": torch.randn(100),
        "matrix": torch.randn(50, 50),
        "tensor_3d": torch.randn(10, 20, 30),
        "tensor_4d": torch.randn(2, 3, 4, 5),
    }
