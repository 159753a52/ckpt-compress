"""
校正模块。

提供幂律拟合、逆映射和评估指标计算功能，
用于建立剪枝预测信号 (x = p/L0) 与实际相对损失 (y = Δloss/L0) 之间的映射关系。
"""

import math
import numpy as np
from typing import List, Tuple, Dict
from scipy import stats


def fit_powerlaw(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """
    拟合幂律模型 y = a * x^b。

    使用对数空间线性回归：log(y) = log(a) + b * log(x)

    参数:
        xs: x 值列表（预测信号 p/L0）
        ys: y 值列表（实际相对损失 Δloss/L0）

    返回:
        (a, b) 参数元组

    异常:
        ValueError: 如果输入无效（空、长度不匹配、包含非正值等）
    """
    # 输入验证
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Input lists cannot be empty")

    if len(xs) != len(ys):
        raise ValueError(f"Length mismatch: xs has {len(xs)} elements, ys has {len(ys)}")

    if len(xs) < 2:
        raise ValueError("Need at least 2 data points for fitting")

    # 转换为 numpy 数组
    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)

    # 检查非正值
    if np.any(xs_arr <= 0):
        raise ValueError("All x values must be positive for log-space fitting")

    if np.any(ys_arr <= 0):
        raise ValueError("All y values must be positive for log-space fitting")

    # 对数变换
    log_x = np.log(xs_arr)
    log_y = np.log(ys_arr)

    # 线性回归: log(y) = log(a) + b * log(x)
    # 使用 numpy 的 polyfit (degree=1)
    b, log_a = np.polyfit(log_x, log_y, 1)
    a = np.exp(log_a)

    return float(a), float(b)


def invert_powerlaw(y_target: float, a: float, b: float) -> float:
    """
    幂律逆映射：x* = (y* / a)^(1/b)

    给定目标相对损失 y*，计算对应的预测信号 x*。

    参数:
        y_target: 目标相对损失
        a: 幂律参数 a
        b: 幂律参数 b

    返回:
        对应的 x 值

    异常:
        ValueError: 如果参数无效
    """
    # 参数验证
    if a <= 0:
        raise ValueError(f"Parameter a must be positive, got {a}")

    if b == 0:
        raise ValueError("Parameter b cannot be zero")

    # 处理零目标
    if y_target == 0:
        return 0.0

    # 处理负目标（理论上不应该出现，但为了鲁棒性）
    if y_target < 0:
        # 返回 0，表示不需要剪枝
        return 0.0

    # 逆映射: x = (y/a)^(1/b)
    x = (y_target / a) ** (1.0 / b)

    return float(x)


def compute_calibration_metrics(
    xs: List[float],
    ys: List[float],
    a: float,
    b: float
) -> Dict[str, float]:
    """
    计算校正模型的评估指标。

    参数:
        xs: x 值列表
        ys: y 值列表（实际值）
        a: 幂律参数 a
        b: 幂律参数 b

    返回:
        包含以下键的字典:
        - r_squared: R² 决定系数
        - rmse: 均方根误差
        - mae: 平均绝对误差
        - spearman: Spearman 相关系数

    异常:
        ValueError: 如果输入无效
    """
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Input lists cannot be empty")

    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)

    # 计算预测值
    y_pred = a * (xs_arr ** b)

    # R² 决定系数
    ss_res = np.sum((ys_arr - y_pred) ** 2)
    ss_tot = np.sum((ys_arr - np.mean(ys_arr)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # RMSE
    rmse = np.sqrt(np.mean((ys_arr - y_pred) ** 2))

    # MAE
    mae = np.mean(np.abs(ys_arr - y_pred))

    # Spearman 相关系数
    if len(xs_arr) >= 2:
        spearman_corr, _ = stats.spearmanr(xs_arr, ys_arr)
    else:
        spearman_corr = 0.0

    return {
        'r_squared': float(r_squared),
        'rmse': float(rmse),
        'mae': float(mae),
        'spearman': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
    }


def compute_calibration_variables(
    sum_pruned_scores: float,
    baseline_loss: float,
    actual_loss_increase: float
) -> Dict[str, float]:
    """
    计算校正变量 x 和 y。

    参数:
        sum_pruned_scores: 被剪枝参数的重要性得分之和 (p)
        baseline_loss: 原始模型损失 (L0)
        actual_loss_increase: 实际损失增量 (Δloss = L1 - L0)

    返回:
        包含以下键的字典:
        - x: p / L0（预测信号）
        - y: Δloss / L0（实际相对损失）
    """
    eps = 1e-8

    x = sum_pruned_scores / (baseline_loss + eps)
    y = actual_loss_increase / (baseline_loss + eps)

    return {
        'x': float(x),
        'y': float(y)
    }
