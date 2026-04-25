"""
Inshrinkerator 指标模块。

实现参数重要性指标:
- 幅度: I_m(w) = |w|
- 敏感度: I_s(w) = |∇L(w) · w|（一阶泰勒近似）
"""

import torch


def magnitude(w: torch.Tensor) -> torch.Tensor:
    """
    计算幅度重要性指标。

    I_m(w) = |w|

    参数:
        w: 权重张量

    返回:
        幅度张量（与 w 形状相同）
    """
    return torch.abs(w)


def sensitivity(w: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    使用一阶泰勒近似计算敏感度重要性指标。

    I_s(w) = |∇L(w) · w|

    参数:
        w: 权重张量
        grad: 梯度张量（与 w 形状相同）

    返回:
        敏感度张量（与 w 形状相同）
    """
    return torch.abs(grad * w)
