"""
ExCP 联合剪枝模块。

实现 ExCP 论文中的权重-动量联合收缩:
- 动量到权重: 使用二阶动量确定权重剪枝阈值
- 权重到动量: 如果权重被剪枝，其动量也必须被剪枝
- 残差百分位剪枝
"""

import torch
from typing import Tuple

# 避免除零的小 epsilon
EPS = 1e-8


def compute_weight_threshold(
    W: torch.Tensor,
    m_t: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    使用二阶动量计算权重剪枝阈值。

    r_w = alpha / sqrt(m_t) * median(|W|)

    参数:
        W: 权重张量（或残差权重）
        m_t: 二阶动量张量（与 W 形状相同或可广播）
        alpha: 控制阈值敏感度的超参数

    返回:
        阈值张量 r_w（与 m_t 形状相同）
    """
    median_W = torch.median(torch.abs(W))
    r_w = alpha / (torch.sqrt(m_t) + EPS) * median_W
    return r_w


def compute_weight_mask(
    W: torch.Tensor,
    m_t: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    基于阈值计算权重掩码。

    M_w(i) = 1 如果 |W(i)| > r_w(i)，否则为 0

    参数:
        W: 权重张量
        m_t: 二阶动量张量
        alpha: 超参数

    返回:
        二值掩码张量（1 = 保留，0 = 剪枝）
    """
    r_w = compute_weight_threshold(W, m_t, alpha)
    # 如果需要，将 r_w 广播到与 W 相同的形状
    if r_w.shape != W.shape:
        r_w = r_w.expand_as(W)
    mask = (torch.abs(W) > r_w).long()
    return mask


def compute_momentum_threshold(
    v_t: torch.Tensor,
    beta: float
) -> torch.Tensor:
    """
    计算动量剪枝阈值。

    r_o = beta * mean(|v_t|)

    参数:
        v_t: 一阶动量张量
        beta: 超参数

    返回:
        标量阈值 r_o
    """
    r_o = beta * torch.mean(torch.abs(v_t))
    return r_o


def compute_momentum_mask(
    v_t: torch.Tensor,
    M_w: torch.Tensor,
    beta: float
) -> torch.Tensor:
    """
    基于阈值和权重掩码计算动量掩码。

    M_o(i) = 1 如果 |v_t(i)| > r_o 且 M_w(i) = 1，否则为 0

    参数:
        v_t: 一阶动量张量
        M_w: 权重掩码张量
        beta: 超参数

    返回:
        二值掩码张量（1 = 保留，0 = 剪枝）
    """
    r_o = compute_momentum_threshold(v_t, beta)
    # M_o = (|v_t| > r_o) AND M_w
    mask = ((torch.abs(v_t) > r_o) & (M_w == 1)).long()
    return mask


def prune_residual_by_percentile(
    dW: torch.Tensor,
    p: float
) -> torch.Tensor:
    """
    按百分位剪枝残差权重。

    dW*(i) = 0 如果 |dW(i)| < |dW| 的第 p 百分位，否则为 dW(i)

    参数:
        dW: 残差权重张量
        p: 百分位（0-100），低于此值的将被剪枝

    返回:
        剪枝后的残差张量
    """
    if p <= 0:
        return dW.clone()

    abs_dW = torch.abs(dW)

    if p >= 100:
        # 除最大值外全部剪枝
        threshold = abs_dW.max()
    else:
        # 计算百分位阈值
        threshold = torch.quantile(abs_dW.flatten().float(), p / 100.0)

    # 创建掩码并应用
    mask = abs_dW >= threshold
    dW_star = dW.clone()
    dW_star[~mask] = 0.0

    return dW_star


def joint_prune(
    dW: torch.Tensor,
    v_t: torch.Tensor,
    m_t: torch.Tensor,
    alpha: float,
    beta: float,
    p: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    执行残差权重和优化器动量的联合剪枝。

    步骤:
    1. 使用二阶动量计算权重掩码（公式 4）
    2. 使用一阶动量和权重掩码计算动量掩码（公式 5）
    3. 对残差应用百分位剪枝（公式 6）
    4. 将掩码应用于 dW 和 v_t

    参数:
        dW: 残差权重张量
        v_t: 一阶动量张量
        m_t: 二阶动量张量
        alpha: 权重阈值超参数
        beta: 动量阈值超参数
        p: 残差剪枝的百分位（0-100）

    返回:
        (pruned_dW, pruned_v_t) 元组
    """
    # 步骤 1: 计算权重掩码
    M_w = compute_weight_mask(dW, m_t, alpha)

    # 步骤 2: 计算动量掩码（依赖于权重掩码）
    M_o = compute_momentum_mask(v_t, M_w, beta)

    # 步骤 3: 对残差应用百分位剪枝
    dW_star = prune_residual_by_percentile(dW, p)

    # 步骤 4: 将权重掩码应用于残差
    dW_star = dW_star * M_w.float()

    # 步骤 5: 将动量掩码应用于优化器状态
    v_star = v_t * M_o.float()

    return dW_star, v_star
