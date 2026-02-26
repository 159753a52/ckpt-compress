"""
自适应剪枝辅助模块。

本模块提供自适应剪枝所需的辅助函数，包括：
- 偏置参数保护（bias preservation）
- 分层分布拟合（per-layer distribution fitting）
- 校准信号处理（calibration signal handling）
- 剪枝预算计算（pruning budget computation）

核心流程：
1. 过滤可剪枝层（保护 bias 参数）
2. 对每层重要性分数拟合概率分布（Weibull）
3. 构建分布参数供优化器使用
4. 根据校准曲线计算目标剪枝预算
5. 调整 epsilon 参数以达到目标损失
"""

import os
import re
from typing import Dict, List, Tuple

import torch

from .calibration import compute_calibration_variables, invert_powerlaw
from .distribution import get_best_fit
from .distribution_fast import fit_distributions_per_layer_fast
from .layer_pruning import convert_scipy_params


def is_bias_param(param_name: str) -> bool:
    """
    检查参数名是否是偏置参数（bias）。

    偏置参数通常不应被剪枝，因为：
    1. 偏置参数数量相对较少，剪枝收益有限
    2. 偏置参数对模型输出有直接影响，剪枝可能导致较大性能损失

    参数:
        param_name: 参数名称，如 "transformer.h.0.attn.c_attn.bias"

    返回:
        bool: 如果参数名以 "bias" 结尾则返回 True

    示例:
        >>> is_bias_param("layer.weight")
        False
        >>> is_bias_param("layer.bias")
        True
        >>> is_bias_param("transformer.h.0.mlp.c_fc.bias")
        True
    """
    if not param_name:
        return False
    # 取参数名的最后一部分（按 "." 分割），转小写后判断是否为 "bias"
    return param_name.lower().split(".")[-1] == "bias"


def filter_prunable_layers(
    layer_scores: Dict[str, torch.Tensor],
    preserve_bias: bool = True
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    过滤出可剪枝的层，将偏置层分离出来。

    在自适应剪枝中，我们通常希望保护偏置参数不被剪枝。
    此函数将输入的层分数字典分为两部分：
    1. 可剪枝层（权重参数）
    2. 冻结层（偏置参数，不参与剪枝）

    参数:
        layer_scores: 每层的重要性分数字典
                     格式: {层名称: 重要性分数张量}
        preserve_bias: 是否保护偏置参数
                      True: 偏置参数不参与剪枝
                      False: 所有参数都可剪枝

    返回:
        Tuple[Dict, List]:
            - prunable: 可剪枝层的分数字典
            - frozen: 被冻结（不剪枝）的层名称列表

    示例:
        >>> scores = {
        ...     "layer.weight": torch.randn(100),
        ...     "layer.bias": torch.randn(10)
        ... }
        >>> prunable, frozen = filter_prunable_layers(scores, preserve_bias=True)
        >>> print(list(prunable.keys()))  # ['layer.weight']
        >>> print(frozen)  # ['layer.bias']
    """
    # 如果不保护偏置，直接返回所有层
    if not preserve_bias:
        return dict(layer_scores), []

    prunable = {}
    frozen = []

    for name, scores in layer_scores.items():
        if is_bias_param(name):
            # 偏置参数加入冻结列表
            frozen.append(name)
        else:
            # 非偏置参数加入可剪枝字典
            prunable[name] = scores

    return prunable, frozen


def fit_distributions_per_layer(
    layer_scores: Dict[str, torch.Tensor],
    min_samples: int = 5,
    use_fast: bool = True,
    max_samples: int = 50000
) -> Dict[str, Dict]:
    """
    对每层的重要性分数拟合概率分布。

    自适应剪枝的核心思想是：每层的重要性分数服从某种概率分布（如 Weibull 分布）。
    通过拟合分布参数，我们可以：
    1. 预测给定阈值下会剪枝多少参数
    2. 使用拉格朗日优化分配各层的剪枝比例

    本函数支持两种拟合方法：
    1. 快速方法（use_fast=True）：使用线性回归拟合 Weibull 分布
       - 优点：速度快（10-100倍提升），GPU 友好
       - 原理：对 Weibull CDF 取对数后变为线性关系
    2. 慢速方法（use_fast=False）：使用 scipy 的 MLE 拟合
       - 优点：更精确，支持多种分布类型
       - 缺点：速度慢，不支持 GPU

    参数:
        layer_scores: 每层的重要性分数字典
                     格式: {层名称: 重要性分数张量}
        min_samples: 最小样本数（仅用于慢速方法）
                    样本数少于此值的层会跳过拟合
        use_fast: 是否使用快速拟合方法
                 True: 使用线性回归法（推荐）
                 False: 使用 scipy MLE 法
        max_samples: 快速方法的最大采样数
                    对于大层，随机采样以加速拟合

    返回:
        Dict[str, Dict]: 每层的拟合结果
            格式: {
                层名称: {
                    "name": 分布名称（如 "weibull"）,
                    "params": 分布参数,
                    "ks_stat": KS 统计量（拟合优度）,
                    ...
                }
            }

    示例:
        >>> scores = {"layer1": torch.rand(1000), "layer2": torch.rand(2000)}
        >>> results = fit_distributions_per_layer(scores, use_fast=True)
        >>> print(results["layer1"]["name"])  # "weibull"
    """
    if use_fast:
        # 使用快速线性回归方法（GPU 友好，速度提升 10-100 倍）
        # 原理：Weibull CDF: F(x) = 1 - exp(-(x/β)^k)
        # 取对数：ln(-ln(1-F(x))) = k*ln(x) - k*ln(β)
        # 这是关于 ln(x) 的线性方程，可用线性回归求解 k 和 β
        return fit_distributions_per_layer_fast(
            layer_scores,
            max_samples=max_samples,
            compute_ks=True  # 计算 KS 统计量以评估拟合质量
        )

    # 原始的 scipy 方法（慢，但更精确）
    # 使用最大似然估计（MLE）拟合多种分布，选择最佳拟合
    fit_results: Dict[str, Dict] = {}

    for layer_name, scores in layer_scores.items():
        # 将 PyTorch 张量转换为 NumPy 数组
        scores_np = scores.detach().cpu().numpy().flatten()

        # 尝试拟合多种分布，返回最佳拟合结果
        best_fit = get_best_fit(scores_np, min_samples=min_samples)

        if best_fit is None:
            # 拟合失败，返回错误信息
            fit_results[layer_name] = {
                "name": "unknown",
                "ks_stat": 1.0,  # KS 统计量为 1 表示最差拟合
                "error": "fitting failed",
            }
        else:
            fit_results[layer_name] = best_fit

    return fit_results


def build_layer_dist_params(
    layer_scores: Dict[str, torch.Tensor],
    fit_results: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    构建分层剪枝优化器所需的分布参数。

    此函数将拟合结果转换为优化器可用的格式。
    优化器需要知道每层的：
    1. 参数数量 N
    2. 分布类型（如 weibull）
    3. 分布参数（k, β, loc）

    Weibull 分布参数说明：
    - k (shape): 形状参数，控制分布的形状
      - k < 1: 分布右偏，大量小值
      - k = 1: 退化为指数分布
      - k > 1: 分布左偏，接近正态分布
    - β (scale): 尺度参数，控制分布的宽度
    - loc: 位置参数，分布的起始位置

    参数:
        layer_scores: 每层的重要性分数字典
                     用于获取每层的参数数量
        fit_results: fit_distributions_per_layer 的返回结果
                    包含每层的分布拟合信息

    返回:
        Dict[str, Dict]: 每层的分布参数
            格式: {
                层名称: {
                    "N": 参数数量,
                    "dist_type": 分布类型,
                    "k": 形状参数,
                    "beta": 尺度参数,
                    "loc": 位置参数
                }
            }

    示例:
        >>> scores = {"layer1": torch.rand(1000)}
        >>> fit_results = fit_distributions_per_layer(scores)
        >>> params = build_layer_dist_params(scores, fit_results)
        >>> print(params["layer1"])
        # {'N': 1000, 'dist_type': 'weibull', 'k': 1.2, 'beta': 0.5, 'loc': 0.0}
    """
    layer_dist_params: Dict[str, Dict] = {}

    for layer_name, scores in layer_scores.items():
        # 获取该层的拟合信息
        fit_info = fit_results.get(layer_name, {})
        dist_name = fit_info.get("name", "weibull")
        params = fit_info.get("params")

        # 尝试将 scipy 格式的参数转换为优化器格式
        converted_params = None
        if params is not None:
            try:
                # scipy 的参数格式与我们的优化器格式不同
                # 例如 scipy 的 weibull_min 使用 (c, loc, scale)
                # 我们需要转换为 (k, beta, loc)
                converted_params = convert_scipy_params(dist_name, params)
            except Exception:
                converted_params = None

        # 如果转换失败，使用默认参数
        if not converted_params:
            # 默认使用 k=1（指数分布），β=0.1，loc=0
            converted_params = {"k": 1.0, "beta": 0.1, "loc": 0.0}
            dist_name = "weibull"

        # 构建该层的完整参数字典
        layer_dist_params[layer_name] = {
            "N": int(scores.numel()),  # 参数数量
            "dist_type": dist_name,     # 分布类型
            **converted_params,         # 分布参数（k, beta, loc）
        }

    return layer_dist_params


def apply_prune_ratio_overrides(
    prune_ratios: Dict[str, float],
    layer_scores: Dict[str, torch.Tensor],
    preserve_bias: bool = True
) -> Dict[str, float]:
    """
    应用剪枝比例覆盖，确保每层都有剪枝比例且偏置被保护。

    此函数用于后处理优化器输出的剪枝比例：
    1. 确保所有层都有剪枝比例（缺失的默认为 0）
    2. 如果需要保护偏置，将偏置层的剪枝比例设为 0

    参数:
        prune_ratios: 优化器输出的剪枝比例字典
                     格式: {层名称: 剪枝比例}
        layer_scores: 所有层的重要性分数字典
                     用于获取完整的层列表
        preserve_bias: 是否保护偏置参数
                      True: 偏置层剪枝比例强制为 0

    返回:
        Dict[str, float]: 完整的剪枝比例字典
            所有层都有对应的剪枝比例

    示例:
        >>> ratios = {"layer.weight": 0.3}
        >>> scores = {"layer.weight": torch.rand(100), "layer.bias": torch.rand(10)}
        >>> result = apply_prune_ratio_overrides(ratios, scores, preserve_bias=True)
        >>> print(result)
        # {'layer.weight': 0.3, 'layer.bias': 0.0}
    """
    # 复制输入字典，转换为 float 类型
    ratios = {name: float(value) for name, value in prune_ratios.items()}

    # 遍历所有层，确保每层都有剪枝比例
    for layer_name in layer_scores.keys():
        if preserve_bias and is_bias_param(layer_name):
            # 偏置层强制不剪枝
            ratios[layer_name] = 0.0
        else:
            # 非偏置层，如果没有指定比例则默认为 0
            ratios.setdefault(layer_name, 0.0)

    return ratios


def compute_target_pruned_score_budget(
    target_rel_loss: float,
    baseline_loss: float,
    powerlaw_params: Dict[str, float]
) -> float:
    """
    根据目标相对损失计算剪枝分数预算 p*。

    自适应剪枝的核心假设是：剪枝导致的损失增加与剪枝分数总和之间
    存在幂律关系：
        y = a * x^b
    其中：
        - x = p / L0（剪枝分数总和 / 基准损失）
        - y = ΔL / L0（损失增加 / 基准损失）
        - a, b 是通过校准实验拟合的参数

    给定目标相对损失 y_target，我们可以反推出允许的剪枝分数预算：
        x_target = (y_target / a)^(1/b)
        p_target = x_target * L0

    参数:
        target_rel_loss: 目标相对损失增加
                        例如 0.001 表示允许 0.1% 的损失增加
        baseline_loss: 基准损失（未剪枝模型的损失）
        powerlaw_params: 幂律参数字典
                        格式: {"a": float, "b": float}

    返回:
        float: 剪枝分数预算 p*
              这是允许剪枝的重要性分数总和上限

    异常:
        ValueError: 如果 powerlaw_params 缺少必要参数

    示例:
        >>> params = {"a": 0.5, "b": 1.2}
        >>> budget = compute_target_pruned_score_budget(0.001, 2.5, params)
        >>> print(f"剪枝预算: {budget:.6f}")
    """
    # 基准损失必须为正
    if baseline_loss <= 0:
        return 0.0

    # 检查幂律参数是否完整
    if not powerlaw_params or "a" not in powerlaw_params or "b" not in powerlaw_params:
        raise ValueError("powerlaw_params must include 'a' and 'b'")

    # 使用幂律反函数计算 x_target
    # x_target = (y_target / a)^(1/b)
    x_target = invert_powerlaw(target_rel_loss, powerlaw_params["a"], powerlaw_params["b"])

    # p_target = x_target * L0
    return float(x_target * baseline_loss)


def collect_calibration_points(
    results: List[Dict],
    baseline_loss: float
) -> Tuple[List[float], List[float]]:
    """
    从校准实验结果中收集 (x, y) 数据点。

    校准实验在不同稀疏度下测量实际损失增加，
    用于拟合幂律关系 y = a * x^b。

    数据点定义：
    - x = sum_pruned_scores / baseline_loss（归一化剪枝分数）
    - y = actual_loss_increase / baseline_loss（归一化损失增加）

    参数:
        results: 校准实验结果列表
                每个元素是一个字典，包含：
                - "sum_pruned_scores": 被剪枝参数的重要性分数总和
                - "actual_loss_increase": 实际损失增加量
        baseline_loss: 基准损失（用于归一化）

    返回:
        Tuple[List[float], List[float]]:
            - xs: x 值列表（归一化剪枝分数）
            - ys: y 值列表（归一化损失增加）
            只包含 x > 0 且 y > 0 的有效点

    示例:
        >>> results = [
        ...     {"sum_pruned_scores": 0.1, "actual_loss_increase": 0.01},
        ...     {"sum_pruned_scores": 0.2, "actual_loss_increase": 0.03},
        ... ]
        >>> xs, ys = collect_calibration_points(results, baseline_loss=2.5)
    """
    xs: List[float] = []
    ys: List[float] = []

    for item in results:
        # 检查必要字段是否存在
        if "sum_pruned_scores" not in item or "actual_loss_increase" not in item:
            continue

        if baseline_loss > 0:
            # 直接计算归一化值
            x = float(item["sum_pruned_scores"]) / baseline_loss
            y = float(item["actual_loss_increase"]) / baseline_loss
        else:
            # 使用辅助函数计算（处理边界情况）
            vars_ = compute_calibration_variables(
                sum_pruned_scores=item["sum_pruned_scores"],
                baseline_loss=baseline_loss,
                actual_loss_increase=item["actual_loss_increase"],
            )
            x = vars_["x"]
            y = vars_["y"]

        # 只保留有效点（x > 0 且 y > 0）
        # 无效点可能来自数值误差或异常情况
        if x > 0 and y > 0:
            xs.append(x)
            ys.append(y)

    return xs, ys


def adjust_epsilon_for_target(
    initial_epsilon: float,
    target_rel_loss: float,
    evaluate_fn,
    max_iters: int = 8
) -> Tuple[float, List[Dict[str, float]]]:
    """
    调整 epsilon 参数以达到目标损失。

    epsilon 是拉格朗日优化中的约束参数，控制总剪枝预算。
    此函数通过二分搜索找到合适的 epsilon，使得：
        actual_rel_loss <= target_rel_loss

    搜索策略：
    1. 从 initial_epsilon 开始
    2. 如果实际损失超过目标，将 epsilon 减半
    3. 重复直到满足目标或达到最大迭代次数

    参数:
        initial_epsilon: 初始 epsilon 值
                        通常由 compute_target_pruned_score_budget 计算得到
        target_rel_loss: 目标相对损失
                        例如 0.001 表示允许 0.1% 的损失增加
        evaluate_fn: 评估函数
                    输入 epsilon，返回实际相对损失
                    签名: (epsilon: float) -> float
        max_iters: 最大迭代次数
                  防止无限循环

    返回:
        Tuple[float, List[Dict]]:
            - epsilon: 最终的 epsilon 值
            - history: 搜索历史，每个元素包含 epsilon 和 actual_rel_loss

    异常:
        ValueError: 如果 initial_epsilon 为负数

    示例:
        >>> def evaluate(eps):
        ...     # 模拟评估函数
        ...     return eps * 10  # 简化示例
        >>> eps, history = adjust_epsilon_for_target(0.1, 0.05, evaluate)
        >>> print(f"最终 epsilon: {eps}")
    """
    if initial_epsilon < 0:
        raise ValueError("initial_epsilon must be non-negative")

    epsilon = float(initial_epsilon)
    history: List[Dict[str, float]] = []

    for _ in range(max_iters):
        # 评估当前 epsilon 下的实际损失
        actual_rel_loss = float(evaluate_fn(epsilon))

        # 记录搜索历史
        history.append({"epsilon": epsilon, "actual_rel_loss": actual_rel_loss})

        # 检查是否满足目标
        if actual_rel_loss <= target_rel_loss:
            break

        # 损失过高，减小 epsilon（更保守的剪枝）
        epsilon *= 0.5

    return epsilon, history


def setup_hf_cache(cache_dir: str) -> None:
    """
    配置 HuggingFace 缓存目录。

    HuggingFace 的模型和数据集会下载到缓存目录。
    此函数设置相关环境变量，确保所有缓存都存储在指定位置。

    设置的环境变量：
    - HF_HOME: HuggingFace 主目录
    - TRANSFORMERS_CACHE: Transformers 模型缓存
    - HF_DATASETS_CACHE: Datasets 数据集缓存

    参数:
        cache_dir: 缓存目录路径
                  如果不存在会自动创建

    示例:
        >>> setup_hf_cache("data/hf_cache")
        # 之后加载模型会使用 data/hf_cache 作为缓存目录
    """
    # 创建缓存目录（如果不存在）
    os.makedirs(cache_dir, exist_ok=True)

    # 设置环境变量
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir


def get_block_number(name: str) -> int:
    """
    从 GPT-2 参数名中提取 Transformer 块索引。

    GPT-2 的参数命名规则：
    - transformer.h.{block_num}.{component}.{param}
    - 例如: transformer.h.0.attn.c_attn.weight

    此函数用于分析不同 Transformer 块的剪枝情况。

    参数:
        name: 参数名称
             例如 "transformer.h.5.mlp.c_fc.weight"

    返回:
        int: 块索引（0-based）
            如果无法提取则返回 -1

    示例:
        >>> get_block_number("transformer.h.0.attn.c_attn.weight")
        0
        >>> get_block_number("transformer.h.11.mlp.c_proj.bias")
        11
        >>> get_block_number("transformer.wte.weight")
        -1  # 嵌入层没有块索引
    """
    if not name:
        return -1

    # 使用正则表达式匹配 ".h.{数字}." 模式
    match = re.search(r"\.h\.(\d+)\.", name)

    if match:
        return int(match.group(1))

    # 无法匹配，返回 -1（表示非 Transformer 块参数）
    return -1
