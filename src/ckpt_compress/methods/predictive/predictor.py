"""
预测残差压缩 - 预测器模块。

实现基于优化器状态的权重预测:
- SGD 预测: W_pred = W - lr * grad
- Adam 预测: W_pred = W - lr * m / (sqrt(v) + eps)

核心思想: 如果我们能准确预测下一个权重，
残差 (W_actual - W_pred) 将会很小，从而更好地压缩。
"""

import torch
from typing import Dict, Optional
from dataclasses import dataclass


def sgd_predict(
    W: torch.Tensor,
    grad: torch.Tensor,
    lr: float
) -> torch.Tensor:
    """
    使用 SGD 更新规则预测下一个权重。

    W_pred = W - lr * grad

    参数:
        W: 当前权重
        grad: 梯度
        lr: 学习率

    返回:
        预测的下一个权重
    """
    return W - lr * grad


def adam_predict(
    W: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    使用 Adam 更新规则预测下一个权重。

    W_pred = W - lr * m / (sqrt(v) + eps)

    参数:
        W: 当前权重
        m: 一阶矩估计 (exp_avg)
        v: 二阶矩估计 (exp_avg_sq)
        lr: 学习率
        eps: 数值稳定性的 epsilon

    返回:
        预测的下一个权重
    """
    return W - lr * m / (torch.sqrt(v) + eps)


def compute_prediction_residual(
    W_actual: torch.Tensor,
    W_pred: torch.Tensor
) -> torch.Tensor:
    """
    计算实际权重和预测权重之间的残差。

    残差 = W_actual - W_pred

    参数:
        W_actual: 实际权重
        W_pred: 预测权重

    返回:
        残差张量
    """
    return W_actual - W_pred


@dataclass
class AdamPredictorConfig:
    """Adam 预测器配置。"""
    lr: float = 0.001
    eps: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999


class AdamPredictor:
    """
    基于 Adam 的权重预测器。

    使用 Adam 优化器状态 (exp_avg, exp_avg_sq) 预测
    下一个权重，从而实现更小的残差用于压缩。
    """

    def __init__(
        self,
        lr: float = 0.001,
        eps: float = 1e-8,
        beta1: float = 0.9,
        beta2: float = 0.999
    ):
        """
        初始化 Adam 预测器。

        参数:
            lr: 学习率
            eps: 数值稳定性的 epsilon
            beta1: 一阶矩衰减率
            beta2: 二阶矩衰减率
        """
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2

    def predict(
        self,
        W_prev: Dict[str, torch.Tensor],
        optimizer_state: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        从前一权重和优化器状态预测下一个权重。

        参数:
            W_prev: 前一权重状态字典
            optimizer_state: 包含 exp_avg 和 exp_avg_sq 的优化器状态字典

        返回:
            预测的权重状态字典
        """
        W_pred = {}

        for key in W_prev:
            W = W_prev[key]

            if key in optimizer_state:
                state = optimizer_state[key]
                m = state.get("exp_avg")
                v = state.get("exp_avg_sq")

                if m is not None and v is not None:
                    W_pred[key] = adam_predict(W, m, v, self.lr, self.eps)
                else:
                    # 无优化器状态，使用恒等预测
                    W_pred[key] = W.clone()
            else:
                # 此键无优化器状态，使用恒等预测
                W_pred[key] = W.clone()

        return W_pred

    def compute_residual(
        self,
        W_actual: Dict[str, torch.Tensor],
        W_prev: Dict[str, torch.Tensor],
        optimizer_state: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        计算用于压缩的预测残差。

        参数:
            W_actual: 实际当前权重
            W_prev: 前一权重
            optimizer_state: 优化器状态字典

        返回:
            残差状态字典
        """
        W_pred = self.predict(W_prev, optimizer_state)
        residual = {}

        for key in W_actual:
            if key in W_pred:
                residual[key] = compute_prediction_residual(
                    W_actual[key], W_pred[key]
                )
            else:
                # 无可用预测，使用直接增量
                if key in W_prev:
                    residual[key] = W_actual[key] - W_prev[key]
                else:
                    residual[key] = W_actual[key].clone()

        return residual

    def reconstruct(
        self,
        residual: Dict[str, torch.Tensor],
        W_prev: Dict[str, torch.Tensor],
        optimizer_state: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        从残差重建实际权重。

        W_actual = W_pred + residual

        参数:
            residual: 残差状态字典
            W_prev: 前一权重
            optimizer_state: 优化器状态字典

        返回:
            重建的权重状态字典
        """
        W_pred = self.predict(W_prev, optimizer_state)
        W_reconstructed = {}

        for key in residual:
            if key in W_pred:
                W_reconstructed[key] = W_pred[key] + residual[key]
            else:
                # 无预测，残差即为实际值
                W_reconstructed[key] = residual[key].clone()

        return W_reconstructed
