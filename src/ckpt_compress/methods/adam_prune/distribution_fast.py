"""
快速分布拟合模块 (GPU 友好)。

使用线性回归法拟合韦伯分布，避免 scipy 的数值优化，
大幅提升拟合速度，支持 GPU 加速。

主要函数：
- fit_weibull_linear_regression: 韦伯分布线性回归拟合
- ks_test_gpu: GPU 上的 KS 检验
- fit_distributions_per_layer_fast: 快速多层分布拟合
"""

import torch
from typing import Dict, Tuple, Callable, Optional, Any


def weibull_cdf_gpu(x: torch.Tensor, k: float, lam: float) -> torch.Tensor:
    """
    韦伯分布的 CDF (GPU 友好)。

    F(x; k, λ) = 1 - exp(-(x/λ)^k)

    参数:
        x: 输入值 (tensor)
        k: 形状参数 (shape)
        lam: 尺度参数 (scale)

    返回:
        CDF 值 (tensor)
    """
    # 处理 x <= 0 的情况
    result = torch.zeros_like(x)
    positive_mask = x > 0
    if positive_mask.any():
        x_pos = x[positive_mask]
        result[positive_mask] = 1 - torch.exp(-((x_pos / lam) ** k))
    return result


def exponential_cdf_gpu(x: torch.Tensor, lam: float) -> torch.Tensor:
    """
    指数分布的 CDF (GPU 友好)。

    F(x; λ) = 1 - exp(-λx)

    参数:
        x: 输入值 (tensor)
        lam: 率参数 (rate)

    返回:
        CDF 值 (tensor)
    """
    result = torch.zeros_like(x)
    positive_mask = x > 0
    if positive_mask.any():
        result[positive_mask] = 1 - torch.exp(-lam * x[positive_mask])
    return result


def fit_weibull_linear_regression(
    data: torch.Tensor,
    eps: float = 1e-10
) -> Tuple[float, float]:
    """
    韦伯分布的线性回归拟合 (GPU 友好)。

    利用韦伯分布 CDF 的性质：
    F(x) = 1 - exp(-(x/λ)^k)

    取两次对数：
    log(-log(1-F(x))) = k·log(x) - k·log(λ)

    令 Y = log(-log(1-F)), X = log(x)
    则 Y = k·X + b, 其中 b = -k·log(λ)

    时间复杂度: O(n log n)，主要是排序

    参数:
        data: 输入数据 (正值)
        eps: 数值稳定性常数

    返回:
        (k, lam) 元组
        - k: 形状参数 (shape)
        - lam: 尺度参数 (scale)
    """
    # 确保数据是一维的
    data = data.flatten()

    # 过滤非正值
    data = data[data > eps]

    if data.numel() < 2:
        # 数据不足，返回默认值
        return 1.0, float(data.mean()) if data.numel() > 0 else 1.0

    # 排序
    sorted_data, _ = torch.sort(data)
    n = sorted_data.numel()

    # 经验 CDF: F_i = (i - 0.3) / (n + 0.4) (中位数秩，Bernard's approximation)
    # 这比 i/n 或 (i-0.5)/n 更准确
    i = torch.arange(1, n + 1, device=data.device, dtype=data.dtype)
    F = (i - 0.3) / (n + 0.4)

    # 确保 F 在 (0, 1) 范围内，避免 log(0) 或 log(负数)
    F = torch.clamp(F, eps, 1 - eps)

    # 变换
    X = torch.log(sorted_data)
    Y = torch.log(-torch.log(1 - F))

    # 过滤无效值 (inf, nan)
    valid_mask = torch.isfinite(X) & torch.isfinite(Y)
    X = X[valid_mask]
    Y = Y[valid_mask]

    if X.numel() < 2:
        return 1.0, float(data.mean())

    # 线性回归: Y = k*X + b
    X_mean = X.mean()
    Y_mean = Y.mean()

    # 计算斜率 k
    numerator = ((X - X_mean) * (Y - Y_mean)).sum()
    denominator = ((X - X_mean) ** 2).sum()

    if denominator.abs() < eps:
        # 数据几乎是常数
        return 1.0, float(data.mean())

    k = numerator / denominator

    # 计算截距 b
    b = Y_mean - k * X_mean

    # 从 b = -k*log(λ) 解出 λ
    # λ = exp(-b/k)
    if k.abs() < eps:
        lam = float(data.mean())
    else:
        lam = torch.exp(-b / k)

    # 确保参数为正
    k = max(float(k.abs()), eps)
    lam = max(float(lam), eps)

    return k, lam


def ks_test_gpu(
    data: torch.Tensor,
    cdf_fn: Callable[[torch.Tensor], torch.Tensor],
    eps: float = 1e-10
) -> float:
    """
    GPU 上的 Kolmogorov-Smirnov 检验。

    KS 统计量 = max|F_n(x) - F(x)|
    其中 F_n 是经验 CDF，F 是理论 CDF

    时间复杂度: O(n log n)，主要是排序

    参数:
        data: 输入数据 (tensor)
        cdf_fn: 理论 CDF 函数，接受 tensor 返回 tensor
        eps: 数值稳定性常数

    返回:
        KS 统计量 (float)
    """
    data = data.flatten()
    data = data[data > eps]

    if data.numel() < 2:
        return 0.0

    n = data.numel()

    # 排序
    sorted_data, _ = torch.sort(data)

    # 经验 CDF: F_n(x_i) = i/n
    empirical_cdf = torch.arange(1, n + 1, device=data.device, dtype=data.dtype) / n

    # 理论 CDF
    theoretical_cdf = cdf_fn(sorted_data)

    # 确保 CDF 值在 [0, 1] 范围内
    theoretical_cdf = torch.clamp(theoretical_cdf, 0, 1)

    # KS 统计量: max|F_n - F|
    # 需要考虑两种情况: F_n(x_i) 和 F_n(x_{i-1})
    diff1 = torch.abs(empirical_cdf - theoretical_cdf)

    # F_n(x_{i-1}) = (i-1)/n
    empirical_cdf_prev = torch.arange(0, n, device=data.device, dtype=data.dtype) / n
    diff2 = torch.abs(empirical_cdf_prev - theoretical_cdf)

    ks_stat = max(diff1.max().item(), diff2.max().item())

    return float(ks_stat)


def fit_distributions_per_layer_fast(
    layer_scores: Dict[str, torch.Tensor],
    max_samples: int = 50000,
    compute_ks: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    快速分布拟合：对每层使用线性回归法拟合韦伯分布。

    相比 scipy 的数值优化，速度提升 10-100 倍。

    参数:
        layer_scores: 每层的重要性分数字典 {layer_name: tensor}
        max_samples: 每层最大采样数（用于下采样加速）
        compute_ks: 是否计算 KS 统计量

    返回:
        拟合结果字典 {layer_name: {name, params, ks_stat}}
        params 格式为 scipy 兼容的 (c, loc, scale) = (k, 0, lam)
    """
    if not layer_scores:
        return {}

    fit_results: Dict[str, Dict[str, Any]] = {}

    for layer_name, scores in layer_scores.items():
        data = scores.flatten()

        # 过滤非正值
        data = data[data > 0]

        if data.numel() < 2:
            # 数据不足
            fit_results[layer_name] = {
                "name": "weibull",
                "params": (1.0, 0.0, 1.0),
                "ks_stat": 1.0,
                "error": "insufficient positive data",
            }
            continue

        # 下采样（如果数据量太大）
        if data.numel() > max_samples:
            indices = torch.randperm(data.numel(), device=data.device)[:max_samples]
            data = data[indices]

        # 线性回归拟合韦伯分布
        try:
            k, lam = fit_weibull_linear_regression(data)

            # 计算 KS 统计量
            ks_stat = 0.0
            if compute_ks:
                ks_stat = ks_test_gpu(data, lambda x: weibull_cdf_gpu(x, k, lam))

            fit_results[layer_name] = {
                "name": "weibull",
                "params": (k, 0.0, lam),  # scipy 格式: (c, loc, scale)
                "ks_stat": ks_stat,
            }
        except Exception as e:
            fit_results[layer_name] = {
                "name": "weibull",
                "params": (1.0, 0.0, float(data.mean())),
                "ks_stat": 1.0,
                "error": str(e),
            }

    return fit_results


def fit_exponential_fast(data: torch.Tensor) -> float:
    """
    指数分布的快速拟合 (解析解)。

    λ = 1 / mean(x)

    参数:
        data: 输入数据

    返回:
        λ 参数
    """
    data = data.flatten()
    data = data[data > 0]

    if data.numel() == 0:
        return 1.0

    mean_val = data.mean()
    if mean_val < 1e-10:
        mean_val = torch.tensor(1e-10, device=data.device, dtype=data.dtype)

    return float(1.0 / mean_val)
