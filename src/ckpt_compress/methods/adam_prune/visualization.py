"""
AdamPrune 分布可视化模块。

提供重要性得分分布的可视化功能，包括：
- 直方图绘制
- 多种分布拟合
- 拟合曲线叠加
- 多层对比网格图
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats


def get_distribution_fits(data: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    对数据拟合多种分布。

    参数:
        data: 输入数据数组

    返回:
        字典，包含每种分布的拟合结果
    """
    # 过滤掉非正值
    data = np.asarray(data).flatten()
    data = data[data > 0]

    if len(data) < 5:
        return {
            'exponential': {'error': 'insufficient data'},
            'lognormal': {'error': 'insufficient data'},
            'weibull': {'error': 'insufficient data'},
            'gamma': {'error': 'insufficient data'},
        }

    results = {}

    distributions = [
        ('exponential', stats.expon, {'floc': 0}),
        ('lognormal', stats.lognorm, {'floc': 0}),
        ('weibull', stats.weibull_min, {'floc': 0}),
        ('gamma', stats.gamma, {'floc': 0}),
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


def fit_and_plot_distribution(
    ax,
    data: np.ndarray,
    layer_name: str,
    n_bins: int = 50
) -> Dict[str, Any]:
    """
    在给定的 axes 上绘制直方图和最佳拟合曲线。

    参数:
        ax: matplotlib axes 对象
        data: 输入数据数组
        layer_name: 层名称（用于标题）
        n_bins: 直方图的 bin 数量

    返回:
        包含最佳拟合信息的字典
    """
    data = np.asarray(data).flatten()
    data_positive = data[data > 0]

    if len(data_positive) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(f'{layer_name}\n(insufficient data)')
        return {'best_dist': None, 'ks_stat': None, 'params': None}

    # 绘制直方图
    ax.hist(data_positive, bins=n_bins, density=True, alpha=0.7,
            color='steelblue', edgecolor='white', label='Data')

    # 拟合分布
    fit_results = get_distribution_fits(data_positive)

    # 找到最佳拟合（KS 统计量最小）
    valid_fits = {k: v for k, v in fit_results.items() if 'ks_stat' in v}

    if not valid_fits:
        ax.set_title(f'{layer_name}\n(fitting failed)')
        return {'best_dist': None, 'ks_stat': None, 'params': None}

    best_name = min(valid_fits.keys(), key=lambda k: valid_fits[k]['ks_stat'])
    best_fit = valid_fits[best_name]

    # 绘制拟合曲线
    x = np.linspace(data_positive.min(), np.percentile(data_positive, 99), 200)

    # 分布名称映射（用于显示）
    dist_display_names = {
        'exponential': 'Exponential',
        'lognormal': 'Log-normal',
        'weibull': 'Weibull',
        'gamma': 'Gamma',
    }

    colors = {
        'exponential': 'red',
        'lognormal': 'green',
        'weibull': 'orange',
        'gamma': 'purple',
    }

    # 绘制最佳拟合曲线（粗线）
    y_best = best_fit['dist'].pdf(x, *best_fit['params'])
    ax.plot(x, y_best, color=colors.get(best_name, 'red'), linewidth=2.5,
            label=f'{dist_display_names.get(best_name, best_name)} (KS={best_fit["ks_stat"]:.4f})')

    # 绘制其他分布曲线（细线，半透明）
    for dist_name, fit_result in valid_fits.items():
        if dist_name != best_name and 'dist' in fit_result:
            try:
                y = fit_result['dist'].pdf(x, *fit_result['params'])
                ax.plot(x, y, color=colors.get(dist_name, 'gray'),
                        linewidth=1, alpha=0.5, linestyle='--',
                        label=f'{dist_display_names.get(dist_name, dist_name)} (KS={fit_result["ks_stat"]:.4f})')
            except Exception:
                pass

    # 计算偏度
    skewness = stats.skew(data_positive)

    # 设置标题和标签
    ax.set_title(f'{layer_name}\nBest: {dist_display_names.get(best_name, best_name)}, '
                 f'KS={best_fit["ks_stat"]:.4f}, Skew={skewness:.2f}')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', fontsize=8)

    # 设置 x 轴为对数刻度（如果数据范围很大）
    if data_positive.max() / (data_positive.min() + 1e-10) > 100:
        ax.set_xscale('log')

    return {
        'best_dist': best_name,
        'ks_stat': best_fit['ks_stat'],
        'params': best_fit['params'],
        'skewness': skewness,
    }


def plot_layer_distributions(
    layer_scores: Dict[str, np.ndarray],
    title: str = "Layer Importance Score Distributions",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> Any:
    """
    绘制所有层的重要性分布图。

    参数:
        layer_scores: 字典，{层名称: 重要性得分数组}
        title: 图表总标题
        save_path: 保存路径（如果为 None 则不保存）
        figsize: 图像大小

    返回:
        matplotlib figure 对象

    异常:
        ValueError: 如果输入为空或包含空数组
    """
    import matplotlib.pyplot as plt

    if not layer_scores:
        raise ValueError("layer_scores cannot be empty")

    # 检查是否有空数组
    for name, scores in layer_scores.items():
        if len(scores) == 0:
            raise ValueError(f"Layer '{name}' has empty scores")

    n_layers = len(layer_scores)

    # 计算网格布局
    if n_layers == 1:
        n_rows, n_cols = 1, 1
    elif n_layers == 2:
        n_rows, n_cols = 1, 2
    elif n_layers <= 4:
        n_rows, n_cols = 2, 2
    elif n_layers <= 6:
        n_rows, n_cols = 2, 3
    elif n_layers <= 9:
        n_rows, n_cols = 3, 3
    else:
        n_cols = 4
        n_rows = (n_layers + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # 确保 axes 是数组
    if n_layers == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    # 绘制每个层
    results = {}
    for idx, (layer_name, scores) in enumerate(sorted(layer_scores.items())):
        ax = axes[idx]
        result = fit_and_plot_distribution(ax, scores, layer_name)
        results[layer_name] = result

    # 隐藏多余的子图
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_comparison_grid(
    layer_scores: Dict[str, np.ndarray],
    title: str = "Layer Distribution Comparison",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> Any:
    """
    创建层分布对比网格图。

    与 plot_layer_distributions 类似，但提供更紧凑的布局。

    参数:
        layer_scores: 字典，{层名称: 重要性得分数组}
        title: 图表总标题
        save_path: 保存路径
        figsize: 图像大小

    返回:
        matplotlib figure 对象
    """
    return plot_layer_distributions(
        layer_scores,
        title=title,
        save_path=save_path,
        figsize=figsize
    )


def plot_marginal_loss_curves(
    marginal_data: Dict[str, Dict[str, List]],
    title: str = "Marginal Loss Curves",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> Any:
    """
    绘制边际损失曲线。

    参数:
        marginal_data: {formula: {'prune_ratios': [...], 'losses': [...]}}
        title: 图表标题
        save_path: 保存路径
        figsize: 图像大小

    返回:
        matplotlib figure 对象

    异常:
        ValueError: 如果输入为空
    """
    import matplotlib.pyplot as plt

    if not marginal_data:
        raise ValueError("marginal_data cannot be empty")

    if figsize is None:
        figsize = (10, 6)

    fig, ax = plt.subplots(figsize=figsize)

    colors = {'A': 'blue', 'B': 'red', 'C': 'green'}
    markers = {'A': 'o', 'B': 's', 'C': '^'}

    for formula, data in sorted(marginal_data.items()):
        prune_ratios = data['prune_ratios']
        losses = data['losses']
        color = colors.get(formula, 'gray')
        marker = markers.get(formula, 'x')

        ax.plot(prune_ratios, losses, color=color, marker=marker,
                linewidth=2, markersize=6, label=f'Formula {formula}')

    ax.set_xlabel('Prune Ratio')
    ax.set_ylabel('Loss Increment')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_prediction_vs_actual(
    results: Dict[str, Dict[str, List]],
    title: str = "Predicted vs Actual Loss",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> Any:
    """
    绘制预测损失 vs 实际损失散点图。

    参数:
        results: {formula: {'predicted': [...], 'actual': [...]}}
        title: 图表标题
        save_path: 保存路径
        figsize: 图像大小

    返回:
        matplotlib figure 对象

    异常:
        ValueError: 如果输入为空
    """
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("results cannot be empty")

    if figsize is None:
        figsize = (8, 8)

    fig, ax = plt.subplots(figsize=figsize)

    colors = {'A': 'blue', 'B': 'red', 'C': 'green'}
    markers = {'A': 'o', 'B': 's', 'C': '^'}

    all_values = []
    for formula, data in sorted(results.items()):
        predicted = data['predicted']
        actual = data['actual']
        all_values.extend(predicted)
        all_values.extend(actual)

        color = colors.get(formula, 'gray')
        marker = markers.get(formula, 'x')

        ax.scatter(predicted, actual, color=color, marker=marker,
                   s=80, alpha=0.7, label=f'Formula {formula}')

    # 绘制对角线（理想预测线）
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        margin = (max_val - min_val) * 0.1
        line_range = [min_val - margin, max_val + margin]
        ax.plot(line_range, line_range, 'k--', linewidth=1, alpha=0.5,
                label='Ideal (y=x)')

    ax.set_xlabel('Predicted Loss')
    ax.set_ylabel('Actual Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_layer_allocation_heatmap(
    allocation_data: Dict[str, Dict[str, float]],
    title: str = "Layer Pruning Allocation",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> Any:
    """
    绘制层剪枝比例热力图。

    参数:
        allocation_data: {formula: {layer_name: prune_ratio}}
        title: 图表标题
        save_path: 保存路径
        figsize: 图像大小

    返回:
        matplotlib figure 对象

    异常:
        ValueError: 如果输入为空
    """
    import matplotlib.pyplot as plt

    if not allocation_data:
        raise ValueError("allocation_data cannot be empty")

    # 收集所有层名称
    all_layers = set()
    for formula_data in allocation_data.values():
        all_layers.update(formula_data.keys())
    layers = sorted(all_layers)
    formulas = sorted(allocation_data.keys())

    # 构建数据矩阵
    data_matrix = np.zeros((len(formulas), len(layers)))
    for i, formula in enumerate(formulas):
        for j, layer in enumerate(layers):
            data_matrix[i, j] = allocation_data[formula].get(layer, 0.0)

    if figsize is None:
        figsize = (max(10, len(layers) * 0.8), max(4, len(formulas) * 1.5))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')

    # 设置刻度
    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(formulas)))
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.set_yticklabels([f'Formula {f}' for f in formulas])

    # 添加数值标注
    for i in range(len(formulas)):
        for j in range(len(layers)):
            value = data_matrix[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=text_color, fontsize=8)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Prune Ratio')

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_formula_comparison(
    comparison_data: Dict[str, Dict[str, float]],
    title: str = "Formula Comparison",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> Any:
    """
    绘制公式对比柱状图。

    参数:
        comparison_data: {formula: {metric_name: value}}
        title: 图表标题
        save_path: 保存路径
        figsize: 图像大小

    返回:
        matplotlib figure 对象

    异常:
        ValueError: 如果输入为空
    """
    import matplotlib.pyplot as plt

    if not comparison_data:
        raise ValueError("comparison_data cannot be empty")

    # 收集所有指标
    all_metrics = set()
    for formula_data in comparison_data.values():
        all_metrics.update(formula_data.keys())
    metrics = sorted(all_metrics)
    formulas = sorted(comparison_data.keys())

    if figsize is None:
        figsize = (max(8, len(metrics) * 2), 6)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(metrics))
    width = 0.8 / len(formulas)
    colors = {'A': 'blue', 'B': 'red', 'C': 'green'}

    for i, formula in enumerate(formulas):
        values = [comparison_data[formula].get(m, 0) for m in metrics]
        offset = (i - len(formulas) / 2 + 0.5) * width
        color = colors.get(formula, 'gray')
        ax.bar(x + offset, values, width, label=f'Formula {formula}',
               color=color, alpha=0.8)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_calibration_scatter(
    xs: List[float],
    ys: List[float],
    a: float,
    b: float,
    title: str = "Calibration: x (p/L0) vs y (Δloss/L0)",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> Any:
    """
    绘制校正散点图（log-log 坐标）+ 拟合曲线。

    参数:
        xs: x 值列表（预测信号 p/L0）
        ys: y 值列表（实际相对损失 Δloss/L0）
        a: 幂律参数 a
        b: 幂律参数 b
        title: 图表标题
        save_path: 保存路径
        figsize: 图像大小

    返回:
        matplotlib figure 对象

    异常:
        ValueError: 如果输入为空
    """
    import matplotlib.pyplot as plt

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("xs and ys cannot be empty")

    if figsize is None:
        figsize = (10, 8)

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制散点
    ax.scatter(xs, ys, color='blue', s=100, alpha=0.7, label='Data points', zorder=3)

    # 绘制拟合曲线
    xs_arr = np.array(xs)
    x_fit = np.linspace(min(xs_arr) * 0.8, max(xs_arr) * 1.2, 100)
    y_fit = a * (x_fit ** b)
    ax.plot(x_fit, y_fit, 'r-', linewidth=2,
            label=f'Fit: y = {a:.4f} * x^{b:.4f}', zorder=2)

    # 设置对数坐标
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('x = p / L0 (Prediction Signal)', fontsize=12)
    ax.set_ylabel('y = Δloss / L0 (Relative Loss)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_sparsity_vs_loss(
    sparsities: List[float],
    y_actuals: List[float],
    title: str = "Sparsity vs Relative Loss",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> Any:
    """
    绘制稀疏度 vs 实际相对损失。

    参数:
        sparsities: 稀疏度列表
        y_actuals: 实际相对损失列表
        title: 图表标题
        save_path: 保存路径
        figsize: 图像大小

    返回:
        matplotlib figure 对象

    异常:
        ValueError: 如果输入为空
    """
    import matplotlib.pyplot as plt

    if len(sparsities) == 0 or len(y_actuals) == 0:
        raise ValueError("sparsities and y_actuals cannot be empty")

    if figsize is None:
        figsize = (10, 6)

    fig, ax = plt.subplots(figsize=figsize)

    # 按稀疏度排序
    sorted_pairs = sorted(zip(sparsities, y_actuals))
    sorted_sparsities = [p[0] for p in sorted_pairs]
    sorted_y_actuals = [p[1] for p in sorted_pairs]

    # 绘制折线图和散点
    ax.plot(sorted_sparsities, sorted_y_actuals, 'b-o', linewidth=2,
            markersize=8, label='Actual relative loss')

    # 将稀疏度转换为百分比显示
    ax.set_xlabel('Global Sparsity (%)', fontsize=12)
    ax.set_ylabel('Relative Loss (Δloss / L0)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # 设置 x 轴为百分比格式
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_calibration_target_vs_actual(
    y_targets: List[float],
    y_actuals: List[float],
    title: str = "Target vs Actual Relative Loss",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> Any:
    """
    绘制目标相对损失 vs 实际相对损失。

    参数:
        y_targets: 目标相对损失列表
        y_actuals: 实际相对损失列表
        title: 图表标题
        save_path: 保存路径
        figsize: 图像大小

    返回:
        matplotlib figure 对象

    异常:
        ValueError: 如果输入为空
    """
    import matplotlib.pyplot as plt

    if len(y_targets) == 0 or len(y_actuals) == 0:
        raise ValueError("y_targets and y_actuals cannot be empty")

    if figsize is None:
        figsize = (8, 8)

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制散点
    ax.scatter(y_targets, y_actuals, color='blue', s=100, alpha=0.7,
               label='Data points', zorder=3)

    # 绘制对角线（理想预测线）
    all_values = list(y_targets) + list(y_actuals)
    min_val = min(all_values)
    max_val = max(all_values)
    margin = (max_val - min_val) * 0.1
    line_range = [min_val - margin, max_val + margin]
    ax.plot(line_range, line_range, 'k--', linewidth=1.5, alpha=0.5,
            label='Ideal (y=x)', zorder=1)

    ax.set_xlabel('Target Relative Loss (y*)', fontsize=12)
    ax.set_ylabel('Actual Relative Loss (y)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
