"""
AdamPrune 分布分析模块。

用于验证假设4：重要性得分是否服从指数分布。
提供分布拟合、统计检验和可视化功能。
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy import stats


def fit_exponential_distribution(
    importance_scores: torch.Tensor
) -> Tuple[float, float]:
    """
    拟合指数分布，返回 (lambda, goodness_of_fit)。

    使用最大似然估计：λ = 1 / mean(x)

    参数:
        importance_scores: 重要性得分张量

    返回:
        (lambda_est, goodness) 元组
        - lambda_est: 估计的 λ 参数
        - goodness: 拟合优度 (基于 KS 检验的 p-value)

    异常:
        ValueError: 如果输入为空
    """
    if importance_scores.numel() == 0:
        raise ValueError("importance_scores cannot be empty")

    data = importance_scores.flatten().numpy()

    # 最大似然估计: λ = 1 / mean
    mean_val = np.mean(data)
    if mean_val < 1e-10:
        mean_val = 1e-10
    lambda_est = 1.0 / mean_val

    # KS 检验评估拟合优度
    if len(data) > 1:
        # 使用 scipy 的 kstest
        ks_stat, p_value = stats.kstest(data, 'expon', args=(0, 1/lambda_est))
        goodness = p_value
    else:
        goodness = 1.0  # 单个值无法进行检验

    return lambda_est, goodness


def analyze_distribution(
    importance_scores: torch.Tensor
) -> Dict[str, Any]:
    """
    分析重要性得分的分布特性。

    参数:
        importance_scores: 重要性得分张量

    返回:
        包含统计量的字典:
        - mean, std, min, max, median: 基本统计量
        - skewness, kurtosis: 高阶矩
        - lambda_exp: 指数分布参数估计
        - ks_statistic, ks_pvalue: KS 检验结果
        - is_exponential: 是否接受指数分布假设

    异常:
        ValueError: 如果输入为空
    """
    if importance_scores.numel() == 0:
        raise ValueError("importance_scores cannot be empty")

    data = importance_scores.flatten().numpy()

    # 基本统计量
    mean_val = float(np.mean(data))
    std_val = float(np.std(data))
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    median_val = float(np.median(data))

    # 高阶矩
    if len(data) > 2:
        skewness = float(stats.skew(data))
        kurtosis = float(stats.kurtosis(data))
    else:
        skewness = 0.0
        kurtosis = 0.0

    # 指数分布拟合
    lambda_exp, _ = fit_exponential_distribution(importance_scores)

    # KS 检验
    if len(data) > 1:
        ks_stat, ks_pvalue = stats.kstest(data, 'expon', args=(0, 1/lambda_exp))
    else:
        ks_stat = 0.0
        ks_pvalue = 1.0

    # 判断是否接受指数分布假设 (α = 0.05)
    is_exponential = ks_pvalue > 0.05

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "median": median_val,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "lambda_exp": lambda_exp,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "is_exponential": is_exponential,
    }


def plot_distribution(
    importance_scores: torch.Tensor,
    title: str = "Importance Score Distribution",
    save_path: Optional[str] = None
) -> Any:
    """
    绘制重要性得分分布图。

    包含:
    - 直方图
    - 拟合的指数分布曲线
    - Q-Q 图

    参数:
        importance_scores: 重要性得分张量
        title: 图表标题
        save_path: 保存路径（如果为 None 则不保存）

    返回:
        matplotlib figure 对象
    """
    import matplotlib.pyplot as plt

    data = importance_scores.flatten().numpy()
    lambda_exp, _ = fit_exponential_distribution(importance_scores)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 直方图 + 拟合曲线
    ax1 = axes[0]
    ax1.hist(data, bins=50, density=True, alpha=0.7, label='Data')

    # 拟合的指数分布
    x = np.linspace(0, np.max(data), 100)
    y = lambda_exp * np.exp(-lambda_exp * x)
    ax1.plot(x, y, 'r-', linewidth=2, label=f'Exp(λ={lambda_exp:.4f})')

    ax1.set_xlabel('Importance Score')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{title}\nHistogram with Exponential Fit')
    ax1.legend()

    # Q-Q 图
    ax2 = axes[1]
    stats.probplot(data, dist='expon', sparams=(0, 1/lambda_exp), plot=ax2)
    ax2.set_title('Q-Q Plot (Exponential)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def exponential_quantile(p: float, lambda_val: float) -> float:
    """
    计算指数分布的 p 分位数。

    F^{-1}(p) = -ln(1-p) / λ

    参数:
        p: 分位数 (0 到 1 之间)
        lambda_val: 指数分布参数 λ

    返回:
        p 分位数的值
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return float('inf')

    return -np.log(1 - p) / lambda_val


def exponential_cdf(x: float, lambda_val: float) -> float:
    """
    计算指数分布的 CDF。

    F(x) = 1 - exp(-λx)

    参数:
        x: 输入值
        lambda_val: 指数分布参数 λ

    返回:
        CDF 值
    """
    if x < 0:
        return 0.0
    return 1 - np.exp(-lambda_val * x)


def fit_multiple_distributions(
    data: np.ndarray,
    min_samples: int = 5
) -> Dict[str, Any]:
    """
    对数据拟合多种分布。

    支持的分布：exponential, lognormal, weibull, gamma, pareto

    参数:
        data: 输入数据数组（numpy array 或 torch.Tensor）
        min_samples: 最小样本数

    返回:
        字典，包含每种分布的拟合结果
        每个分布的结果包含: ks_stat, p_value, params, dist
        如果拟合失败，包含: error
    """
    # 转换为numpy数组
    if hasattr(data, 'numpy'):
        data = data.numpy()
    data = np.asarray(data).flatten()

    # 过滤掉非正值
    data = data[data > 0]

    # 检查数据量
    if len(data) < min_samples:
        return {
            'exponential': {'error': 'insufficient data'},
            'lognormal': {'error': 'insufficient data'},
            'weibull': {'error': 'insufficient data'},
            'gamma': {'error': 'insufficient data'},
            'pareto': {'error': 'insufficient data'},
        }

    results = {}

    distributions = [
        ('exponential', stats.expon, {'floc': 0}),
        ('lognormal', stats.lognorm, {'floc': 0}),
        ('weibull', stats.weibull_min, {'floc': 0}),
        ('gamma', stats.gamma, {'floc': 0}),
        ('pareto', stats.pareto, {'floc': 0}),
    ]

    for name, dist, fit_kwargs in distributions:
        try:
            params = dist.fit(data, **fit_kwargs)
            ks_stat, p_value = stats.kstest(data, dist.cdf, args=params)

            results[name] = {
                'ks_stat': float(ks_stat),
                'p_value': float(p_value),
                'params': params,
                'dist': dist,
            }
        except Exception as e:
            results[name] = {'error': str(e)}

    return results


def get_best_fit(
    data: np.ndarray,
    min_samples: int = 5
) -> Optional[Dict[str, Any]]:
    """
    获取最佳拟合分布（KS统计量最小）。

    参数:
        data: 输入数据数组（numpy array 或 torch.Tensor）
        min_samples: 最小样本数

    返回:
        最佳拟合结果字典，包含:
        - name: 分布名称
        - ks_stat: KS统计量
        - p_value: p值
        - params: 分布参数
        - dist: scipy分布对象
        如果没有有效拟合，返回None
    """
    fit_results = fit_multiple_distributions(data, min_samples)

    # 过滤出有效的拟合结果
    valid_fits = {
        name: result
        for name, result in fit_results.items()
        if 'ks_stat' in result
    }

    if not valid_fits:
        return None

    # 找到KS统计量最小的分布
    best_name = min(valid_fits.keys(), key=lambda k: valid_fits[k]['ks_stat'])
    best_result = valid_fits[best_name]

    return {
        'name': best_name,
        'ks_stat': best_result['ks_stat'],
        'p_value': best_result['p_value'],
        'params': best_result['params'],
        'dist': best_result['dist'],
    }
