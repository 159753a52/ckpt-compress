"""
ExCP 残差计算模块。

实现 ExCP 的残差检查点计算:
ΔW_t = W_t - Ŵ_{t-1}

其中 Ŵ_{t-1} 是重建的（非原始的）前一个检查点。
"""

import torch
from typing import Dict


def compute_residual(W_prev_hat: torch.Tensor, W_t: torch.Tensor) -> torch.Tensor:
    """
    计算当前权重与重建的前一权重之间的残差。

    参数:
        W_prev_hat: 重建的前一权重 (Ŵ_{t-1})
        W_t: 当前权重 (W_t)

    返回:
        残差张量 ΔW_t = W_t - Ŵ_{t-1}
    """
    return W_t - W_prev_hat


def reconstruct(W_prev_hat: torch.Tensor, dW: torch.Tensor) -> torch.Tensor:
    """
    从前一重建权重和残差重建权重。

    参数:
        W_prev_hat: 重建的前一权重 (Ŵ_{t-1})
        dW: 残差 (ΔW_t 或量化后的 Δ̂W_t)

    返回:
        重建的权重 Ŵ_t = Ŵ_{t-1} + ΔW_t
    """
    return W_prev_hat + dW


def compute_residual_state_dict(
    prev_hat: Dict[str, torch.Tensor],
    current: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    计算整个状态字典的残差。

    参数:
        prev_hat: 重建的前一状态字典
        current: 当前状态字典

    返回:
        残差状态字典
    """
    residual = {}
    for key in current:
        if key in prev_hat:
            residual[key] = compute_residual(prev_hat[key], current[key])
        else:
            # 新键，原样存储（完整值）
            residual[key] = current[key].clone()
    return residual


def reconstruct_state_dict(
    prev_hat: Dict[str, torch.Tensor],
    residual: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    从前一重建状态字典和残差重建状态字典。

    参数:
        prev_hat: 重建的前一状态字典
        residual: 残差状态字典

    返回:
        重建的状态字典
    """
    reconstructed = {}
    for key in residual:
        if key in prev_hat:
            reconstructed[key] = reconstruct(prev_hat[key], residual[key])
        else:
            # 新键，直接使用残差
            reconstructed[key] = residual[key].clone()
    return reconstructed
