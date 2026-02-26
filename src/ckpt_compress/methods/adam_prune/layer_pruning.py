"""
分层优化剪枝模块。

基于拉格朗日优化的分层剪枝比例计算。
根据每层的重要性分布，计算最优的剪枝比例分配。
"""

import math
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
from scipy.stats import norm


def convert_scipy_params(dist_type: str, scipy_params: tuple) -> Dict[str, float]:
    """
    将 scipy 拟合参数转换到我们的参数格式。

    参数:
        dist_type: 分布类型 ('weibull', 'lognormal', 'exponential', 'gamma')
        scipy_params: scipy 拟合返回的参数元组

    返回:
        转换后的参数字典

    异常:
        ValueError: 如果分布类型未知
    """
    if dist_type == 'weibull':
        # scipy weibull_min 返回 (c, loc, scale)
        c, loc, scale = scipy_params
        return {
            'k': float(c),
            'beta': float(scale),
            'loc': float(loc)
        }

    elif dist_type == 'lognormal':
        # scipy lognorm 返回 (s, loc, scale)
        s, loc, scale = scipy_params
        return {
            'mu': math.log(float(scale)),
            'sigma': float(s),
            'loc': float(loc)
        }

    elif dist_type == 'exponential':
        # scipy expon 返回 (loc, scale)
        loc, scale = scipy_params
        return {
            'lambda': 1.0 / float(scale),
            'loc': float(loc)
        }

    elif dist_type == 'gamma':
        # scipy gamma 返回 (a, loc, scale)
        a, loc, scale = scipy_params
        return {
            'alpha': float(a),
            'beta': 1.0 / float(scale),
            'loc': float(loc)
        }

    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def compute_weibull_prune_ratio(eta: float, N: int, k: float, beta: float) -> float:
    """
    计算韦伯分布的剪枝比例。

    公式: p = 1 - exp(-(1/(η·N·β))^k)

    参数:
        eta: 拉格朗日乘子
        N: 参数数量
        k: 形状参数
        beta: 尺度参数

    返回:
        剪枝比例 p ∈ [0, 1]
    """
    if eta <= 0 or N <= 0 or beta <= 0:
        return 0.0

    arg = 1.0 / (eta * N * beta)
    # 防止数值溢出
    if arg ** k > 700:  # exp(-700) ≈ 0
        return 1.0
    p = 1.0 - math.exp(-(arg ** k))
    return max(0.0, min(1.0, p))


def compute_lognormal_prune_ratio(eta: float, N: int, mu: float, sigma: float) -> float:
    """
    计算对数正态分布的剪枝比例。

    公式: p = Φ((ln(1/η) - ln(N) - μ) / σ)

    参数:
        eta: 拉格朗日乘子
        N: 参数数量
        mu: 位置参数
        sigma: 尺度参数

    返回:
        剪枝比例 p ∈ [0, 1]
    """
    if eta <= 0 or N <= 0 or sigma <= 0:
        return 0.0

    z = (math.log(1.0 / eta) - math.log(N) - mu) / sigma
    p = norm.cdf(z)
    return max(0.0, min(1.0, float(p)))


def compute_weibull_loss(N: int, k: float, beta: float, p: float) -> float:
    """
    计算韦伯分布的损失。

    使用小 p 近似: ΔL ≈ N·β/(1+1/k) · p^(1+1/k)

    参数:
        N: 参数数量
        k: 形状参数
        beta: 尺度参数
        p: 剪枝比例

    返回:
        损失值
    """
    if p <= 0:
        return 0.0

    exponent = 1.0 + 1.0 / k
    loss = N * beta / exponent * (p ** exponent)
    return float(loss)


def compute_lognormal_loss(N: int, mu: float, sigma: float, p: float) -> float:
    """
    计算对数正态分布的损失。

    公式: ΔL = N·exp(μ + σ²/2)·Φ(Φ⁻¹(p) - σ)

    参数:
        N: 参数数量
        mu: 位置参数
        sigma: 尺度参数
        p: 剪枝比例

    返回:
        损失值
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        p = 0.9999  # 避免 Φ⁻¹(1) = inf

    z = norm.ppf(p)  # Φ⁻¹(p)
    loss = N * math.exp(mu + sigma ** 2 / 2) * norm.cdf(z - sigma)
    return float(loss)


class LayerPruningOptimizer:
    """
    分层剪枝优化器。

    使用简化的优化策略：
    - 根据每层的平均重要性分配剪枝比例
    - 低重要性层剪枝更多，高重要性层剪枝更少
    """

    def __init__(self):
        pass

    def compute_layer_prune_ratios(
        self,
        layer_scores: Dict[str, torch.Tensor],
        epsilon: float
    ) -> Dict[str, float]:
        """
        计算每层的最优剪枝比例。

        使用基于重要性的分配策略：
        - 计算每层的平均重要性
        - 重要性越低的层，剪枝比例越高

        参数:
            layer_scores: 每层的重要性得分 {name: tensor}
            epsilon: 可容忍的总损失增量

        返回:
            每层的剪枝比例 {name: float}

        异常:
            ValueError: 如果 epsilon < 0
        """
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        if epsilon == 0:
            return {name: 0.0 for name in layer_scores}

        # 计算每层的统计信息
        layer_stats = {}
        total_params = 0

        for name, scores in layer_scores.items():
            flat_scores = scores.flatten()
            n_params = flat_scores.numel()
            mean_score = flat_scores.mean().item()
            total_params += n_params

            layer_stats[name] = {
                'n_params': n_params,
                'mean_score': mean_score,
                'scores': flat_scores,
            }

        # 计算全局平均重要性
        total_importance = sum(
            s['n_params'] * s['mean_score'] for s in layer_stats.values()
        )
        global_mean = total_importance / total_params if total_params > 0 else 1.0

        # 使用简化公式计算每层剪枝比例
        # p_layer = base_ratio * (global_mean / layer_mean)
        # 这样低重要性层会有更高的剪枝比例

        # 首先计算基础剪枝比例（基于全局公式）
        # p* = sqrt(2ε / (N * s̄))
        base_ratio = np.sqrt(2 * epsilon / (total_params * global_mean + 1e-10))
        base_ratio = min(base_ratio, 1.0)

        ratios = {}
        for name, stats in layer_stats.items():
            layer_mean = stats['mean_score']
            if layer_mean < 1e-10:
                layer_mean = 1e-10

            # 重要性越低，剪枝比例越高
            ratio = base_ratio * (global_mean / layer_mean)
            ratio = max(0.0, min(1.0, ratio))
            ratios[name] = ratio

        return ratios

    def compute_predicted_loss(
        self,
        layer_scores: Dict[str, torch.Tensor],
        prune_ratios: Dict[str, float]
    ) -> float:
        """
        计算预测的损失增量。

        损失增量 = 被剪枝参数的重要性得分之和

        参数:
            layer_scores: 每层的重要性得分 {name: tensor}
            prune_ratios: 每层的剪枝比例 {name: float}

        返回:
            预测的损失增量
        """
        total_loss = 0.0

        for name, scores in layer_scores.items():
            ratio = prune_ratios.get(name, 0.0)
            if ratio <= 0:
                continue

            flat_scores = scores.flatten()
            n_params = flat_scores.numel()
            n_prune = int(n_params * ratio)

            if n_prune > 0:
                # 排序找到最小的 n_prune 个得分
                sorted_scores, _ = torch.sort(flat_scores)
                pruned_scores = sorted_scores[:n_prune]
                total_loss += pruned_scores.sum().item()

        return total_loss

    def apply_pruning(
        self,
        weights: Dict[str, torch.Tensor],
        importance_scores: Dict[str, torch.Tensor],
        prune_ratios: Dict[str, float]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        应用剪枝，返回剪枝后的权重和掩码。

        参数:
            weights: 模型权重 {name: tensor}
            importance_scores: 重要性得分 {name: tensor}
            prune_ratios: 剪枝比例 {name: float}

        返回:
            (pruned_weights, masks) 元组
        """
        pruned_weights = {}
        masks = {}

        for name, weight in weights.items():
            scores = importance_scores[name]
            ratio = prune_ratios.get(name, 0.0)

            shape = weight.shape
            flat_weight = weight.flatten().clone()
            flat_scores = scores.flatten()

            n_params = flat_weight.numel()
            n_prune = int(n_params * ratio)

            mask = torch.ones_like(flat_weight)

            if n_prune > 0:
                # 找到重要性最低的参数
                _, indices = torch.sort(flat_scores)
                prune_indices = indices[:n_prune]
                mask[prune_indices] = 0
                flat_weight[prune_indices] = 0

            masks[name] = mask.reshape(shape)
            pruned_weights[name] = flat_weight.reshape(shape)

        return pruned_weights, masks

    def compute_actual_loss(
        self,
        importance_scores: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> float:
        """
        计算实际的损失增量（被剪枝参数的重要性之和）。

        参数:
            importance_scores: 重要性得分 {name: tensor}
            masks: 剪枝掩码 {name: tensor} (1=保留, 0=剪枝)

        返回:
            实际损失增量
        """
        total_loss = 0.0

        for name, scores in importance_scores.items():
            if name in masks:
                mask = masks[name]
                # 被剪枝的参数 (mask == 0) 的得分之和
                pruned_scores = scores * (1 - mask)
                total_loss += pruned_scores.sum().item()

        return total_loss

    def compute_global_sparsity(
        self,
        masks: Dict[str, torch.Tensor]
    ) -> float:
        """
        计算全局稀疏度。

        参数:
            masks: 剪枝掩码 {name: tensor} (1=保留, 0=剪枝)

        返回:
            全局稀疏度 (被剪枝参数比例)
        """
        total_params = 0
        total_pruned = 0

        for mask in masks.values():
            total_params += mask.numel()
            total_pruned += (mask == 0).sum().item()

        if total_params == 0:
            return 0.0

        return total_pruned / total_params

    def optimize_with_distributions(
        self,
        epsilon: float,
        layer_dist_params: Dict[str, Dict]
    ) -> Tuple[Dict[str, float], float]:
        """
        基于分布参数优化各层剪枝比例。

        使用二分搜索找到最优拉格朗日乘子 η，使得总损失接近 epsilon。

        参数:
            epsilon: 可容忍的总损失增量
            layer_dist_params: 每层的分布参数 {name: {N, dist_type, ...}}

        返回:
            (ratios, eta) 元组，ratios 是每层的剪枝比例，eta 是最优乘子
        """
        if epsilon <= 0:
            return {name: 0.0 for name in layer_dist_params}, float('inf')

        # 二分搜索 η
        eta_low, eta_high = 1e-15, 1e15

        for _ in range(100):  # 最多迭代 100 次
            eta_mid = math.sqrt(eta_low * eta_high)  # 几何平均

            # 计算当前 η 下的各层剪枝比例
            ratios = self._compute_ratios_for_eta(eta_mid, layer_dist_params)

            # 计算总损失
            total_loss = self.compute_distribution_loss(layer_dist_params, ratios)

            if abs(total_loss - epsilon) < epsilon * 0.01:  # 1% 误差范围
                break

            if total_loss > epsilon:
                eta_low = eta_mid  # 损失太大，增大 η（减少剪枝）
            else:
                eta_high = eta_mid  # 损失有余，减小 η（增加剪枝）

        optimal_eta = math.sqrt(eta_low * eta_high)
        ratios = self._compute_ratios_for_eta(optimal_eta, layer_dist_params)

        return ratios, optimal_eta

    def _compute_ratios_for_eta(
        self,
        eta: float,
        layer_dist_params: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        给定 η，计算各层的剪枝比例。

        参数:
            eta: 拉格朗日乘子
            layer_dist_params: 每层的分布参数

        返回:
            每层的剪枝比例
        """
        ratios = {}

        for name, params in layer_dist_params.items():
            N = params['N']
            dist_type = params['dist_type']

            if dist_type == 'weibull':
                k = params['k']
                beta = params['beta']
                p = compute_weibull_prune_ratio(eta, N, k, beta)
            elif dist_type == 'lognormal':
                mu = params['mu']
                sigma = params['sigma']
                p = compute_lognormal_prune_ratio(eta, N, mu, sigma)
            else:
                # 对于其他分布类型，使用韦伯近似
                k = params.get('k', 1.0)
                beta = params.get('beta', 0.1)
                p = compute_weibull_prune_ratio(eta, N, k, beta)

            ratios[name] = p

        return ratios

    def compute_distribution_loss(
        self,
        layer_dist_params: Dict[str, Dict],
        ratios: Dict[str, float]
    ) -> float:
        """
        基于分布参数计算总损失。

        参数:
            layer_dist_params: 每层的分布参数
            ratios: 每层的剪枝比例

        返回:
            总损失
        """
        total_loss = 0.0

        for name, params in layer_dist_params.items():
            p = ratios.get(name, 0.0)
            if p <= 0:
                continue

            N = params['N']
            dist_type = params['dist_type']

            if dist_type == 'weibull':
                k = params['k']
                beta = params['beta']
                loss = compute_weibull_loss(N, k, beta, p)
            elif dist_type == 'lognormal':
                mu = params['mu']
                sigma = params['sigma']
                loss = compute_lognormal_loss(N, mu, sigma, p)
            else:
                # 对于其他分布类型，使用韦伯近似
                k = params.get('k', 1.0)
                beta = params.get('beta', 0.1)
                loss = compute_weibull_loss(N, k, beta, p)

            total_loss += loss

        return total_loss


def compute_real_loss_increase(
    original_model: torch.nn.Module,
    pruned_model: torch.nn.Module,
    data: list,
    num_batches: int = 5
) -> float:
    """
    计算真实的损失增量。

    通过在验证数据上比较原始模型和剪枝后模型的损失来计算实际损失增量。

    参数:
        original_model: 原始模型
        pruned_model: 剪枝后的模型
        data: 验证数据列表，每个元素是一个字典，包含模型输入
        num_batches: 使用的批次数量

    返回:
        损失增量 (pruned_loss - original_loss)
    """
    original_model.eval()
    pruned_model.eval()

    original_losses = []
    pruned_losses = []

    with torch.no_grad():
        for i, batch in enumerate(data):
            if i >= num_batches:
                break

            # 处理不同的输入格式
            if isinstance(batch, dict):
                # 字典格式：可能是 {'input': tensor, 'labels': tensor}
                # 或者 {'input_ids': tensor, 'labels': tensor}
                if 'input' in batch:
                    inputs = batch['input']
                    labels = batch.get('labels', None)
                    orig_output = original_model(inputs, labels=labels)
                    pruned_output = pruned_model(inputs, labels=labels)
                elif 'input_ids' in batch:
                    orig_output = original_model(**batch)
                    pruned_output = pruned_model(**batch)
                else:
                    # 尝试直接传递字典
                    orig_output = original_model(**batch)
                    pruned_output = pruned_model(**batch)
            else:
                # 假设是元组或其他可迭代对象
                orig_output = original_model(*batch)
                pruned_output = pruned_model(*batch)

            # 获取损失值
            if hasattr(orig_output, 'loss') and orig_output.loss is not None:
                original_losses.append(orig_output.loss.item())
            if hasattr(pruned_output, 'loss') and pruned_output.loss is not None:
                pruned_losses.append(pruned_output.loss.item())

    if not original_losses or not pruned_losses:
        return 0.0

    avg_original_loss = np.mean(original_losses)
    avg_pruned_loss = np.mean(pruned_losses)

    return float(avg_pruned_loss - avg_original_loss)


def compute_relative_loss_metrics(
    original_loss: float,
    actual_loss_increase: float,
    sum_pruned_scores: float,
    total_importance: float
) -> Dict[str, float]:
    """
    计算相对损失指标用于校正。

    将预测损失和实际损失都转换为相对值（无量纲），使其可比。

    参数:
        original_loss: 原始模型损失（baseline）
        actual_loss_increase: 实际损失增量 (L1 - L0)
        sum_pruned_scores: 被剪枝参数的重要性得分之和（预测信号 p）
        total_importance: 所有参数的重要性得分之和

    返回:
        包含以下键的字典:
        - rel_actual: 相对实际损失增量 = actual_loss_increase / original_loss
        - rel_predicted: 相对预测损失 = sum_pruned_scores / total_importance
        - calibration_factor: 校准因子 = rel_actual / rel_predicted
    """
    # 使用小常数避免除零
    eps = 1e-8

    rel_actual = actual_loss_increase / (original_loss + eps)
    rel_predicted = sum_pruned_scores / (total_importance + eps)
    calibration_factor = rel_actual / (rel_predicted + eps)

    return {
        'rel_actual': float(rel_actual),
        'rel_predicted': float(rel_predicted),
        'calibration_factor': float(calibration_factor)
    }


def compute_sum_pruned_scores(
    layer_scores: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor]
) -> float:
    """
    计算被剪枝参数的重要性得分之和（真正的预测信号 p）。

    参数:
        layer_scores: 每层的重要性得分 {name: tensor}
        masks: 剪枝掩码 {name: tensor} (1=保留, 0=剪枝)

    返回:
        被剪枝参数的重要性得分之和
    """
    total = 0.0

    for name, scores in layer_scores.items():
        if name in masks:
            mask = masks[name]
            # 被剪枝的参数 (mask == 0) 的得分之和
            pruned_scores = scores * (1 - mask)
            total += pruned_scores.sum().item()

    return float(total)


def prune_by_global_sparsity(
    weights: Dict[str, torch.Tensor],
    importance_scores: Dict[str, torch.Tensor],
    target_sparsity: float
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    按固定全局稀疏度剪枝（不依赖优化器）。

    根据全局重要性排序，剪枝重要性最低的参数直到达到目标稀疏度。

    参数:
        weights: 模型权重 {name: tensor}
        importance_scores: 重要性得分 {name: tensor}
        target_sparsity: 目标全局稀疏度 (0.0 - 1.0)

    返回:
        (pruned_weights, masks) 元组

    异常:
        ValueError: 如果 target_sparsity 不在 [0, 1] 范围内
    """
    if target_sparsity < 0 or target_sparsity > 1:
        raise ValueError(f"target_sparsity must be in [0, 1], got {target_sparsity}")

    # 收集所有参数的重要性得分和位置信息
    all_scores = []
    all_indices = []  # (layer_name, flat_index)

    for name, scores in importance_scores.items():
        flat_scores = scores.flatten()
        for i, score in enumerate(flat_scores):
            all_scores.append(score.item())
            all_indices.append((name, i))

    total_params = len(all_scores)

    if total_params == 0:
        return {}, {}

    # 计算需要剪枝的参数数量
    n_prune = int(total_params * target_sparsity)

    # 找到剪枝阈值（重要性最低的 n_prune 个参数）
    if n_prune == 0:
        prune_set = set()
    elif n_prune >= total_params:
        prune_set = set(range(total_params))
    else:
        # 按重要性排序，找到最低的 n_prune 个
        sorted_indices = sorted(range(total_params), key=lambda i: all_scores[i])
        prune_set = set(sorted_indices[:n_prune])

    # 构建每层的掩码
    pruned_weights = {}
    masks = {}

    for name, weight in weights.items():
        shape = weight.shape
        flat_weight = weight.flatten().clone()
        mask = torch.ones_like(flat_weight)

        # 找到该层需要剪枝的索引
        for global_idx in prune_set:
            layer_name, local_idx = all_indices[global_idx]
            if layer_name == name:
                mask[local_idx] = 0
                flat_weight[local_idx] = 0

        masks[name] = mask.reshape(shape)
        pruned_weights[name] = flat_weight.reshape(shape)

    return pruned_weights, masks


def prune_by_layer_sparsity(
    weights: Dict[str, torch.Tensor],
    importance_scores: Dict[str, torch.Tensor],
    layer_ratios: Dict[str, float]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    按分层稀疏度剪枝（向量化实现）。

    每层独立剪枝，根据该层的重要性排序，剪枝重要性最低的参数。

    参数:
        weights: 模型权重 {name: tensor}
        importance_scores: 重要性得分 {name: tensor}
        layer_ratios: 每层的剪枝比例 {name: float}，缺失的层默认为 0

    返回:
        (pruned_weights, masks) 元组
    """
    if not weights:
        return {}, {}

    pruned_weights = {}
    masks = {}

    for name, weight in weights.items():
        scores = importance_scores[name]
        ratio = layer_ratios.get(name, 0.0)

        shape = weight.shape
        flat_weight = weight.flatten().clone()
        flat_scores = scores.flatten()
        n_params = flat_weight.numel()
        n_prune = int(n_params * ratio)

        # 创建掩码
        mask = torch.ones_like(flat_weight)

        if n_prune > 0 and n_prune <= n_params:
            # 使用 torch.kthvalue 找到第 k 小的值作为阈值
            # kthvalue 返回 (values, indices)
            threshold, _ = torch.kthvalue(flat_scores, n_prune)
            # 剪枝所有小于等于阈值的参数
            # 为了精确控制剪枝数量，使用 topk
            _, prune_indices = torch.topk(flat_scores, n_prune, largest=False)
            mask[prune_indices] = 0
            flat_weight[prune_indices] = 0

        masks[name] = mask.reshape(shape)
        pruned_weights[name] = flat_weight.reshape(shape)

    return pruned_weights, masks


def prune_by_global_sparsity_fast(
    weights: Dict[str, torch.Tensor],
    importance_scores: Dict[str, torch.Tensor],
    target_sparsity: float
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    按固定全局稀疏度剪枝（向量化快速实现）。

    使用 torch.kthvalue 找到全局阈值，避免 Python 循环。

    参数:
        weights: 模型权重 {name: tensor}
        importance_scores: 重要性得分 {name: tensor}
        target_sparsity: 目标全局稀疏度 (0.0 - 1.0)

    返回:
        (pruned_weights, masks) 元组

    异常:
        ValueError: 如果 target_sparsity 不在 [0, 1] 范围内
    """
    if target_sparsity < 0 or target_sparsity > 1:
        raise ValueError(f"target_sparsity must be in [0, 1], got {target_sparsity}")

    if not weights:
        return {}, {}

    # 收集所有参数的重要性得分
    all_scores_list = []
    layer_info = []  # [(name, start_idx, end_idx, shape), ...]

    current_idx = 0
    for name, scores in importance_scores.items():
        flat_scores = scores.flatten()
        n = flat_scores.numel()
        all_scores_list.append(flat_scores)
        layer_info.append((name, current_idx, current_idx + n, scores.shape))
        current_idx += n

    if current_idx == 0:
        return {}, {}

    # 拼接所有分数
    # 确保所有张量在同一设备上
    device = all_scores_list[0].device
    all_scores = torch.cat([s.to(device) for s in all_scores_list])
    total_params = all_scores.numel()

    # 计算需要剪枝的参数数量
    n_prune = int(total_params * target_sparsity)

    # 找到全局阈值
    if n_prune == 0:
        threshold = float('-inf')
    elif n_prune >= total_params:
        threshold = float('inf')
    else:
        threshold, _ = torch.kthvalue(all_scores, n_prune)
        threshold = threshold.item()

    # 构建每层的掩码
    pruned_weights = {}
    masks = {}

    for name, start_idx, end_idx, shape in layer_info:
        weight = weights[name]
        flat_weight = weight.flatten().clone()
        layer_scores = all_scores[start_idx:end_idx]

        # 创建掩码：小于阈值的被剪枝
        if n_prune == 0:
            mask = torch.ones_like(flat_weight)
        elif n_prune >= total_params:
            mask = torch.zeros_like(flat_weight)
            flat_weight.zero_()
        else:
            # 使用全局 topk 来精确控制剪枝数量
            # 但这里我们需要按层处理，所以使用阈值
            mask = (layer_scores > threshold).float()
            # 处理边界情况：等于阈值的参数
            # 为了精确控制总剪枝数量，我们需要特殊处理
            flat_weight = flat_weight * mask

        masks[name] = mask.reshape(shape)
        pruned_weights[name] = flat_weight.reshape(shape)

    return pruned_weights, masks


def compute_layer_sparsities_from_global(
    layer_stats: Dict[str, Dict],
    target_sparsity: float
) -> Dict[str, float]:
    """
    从全局稀疏度计算每层的剪枝比例。

    基于每层的平均重要性分配剪枝比例：
    - 低重要性层剪枝更多
    - 高重要性层剪枝更少
    - 总剪枝参数数接近全局目标

    参数:
        layer_stats: 每层的统计信息 {name: {n_params, mean_importance}}
        target_sparsity: 目标全局稀疏度 (0.0 - 1.0)

    返回:
        每层的剪枝比例 {name: float}
    """
    if target_sparsity == 0:
        return {name: 0.0 for name in layer_stats}

    if not layer_stats:
        return {}

    # 计算全局平均重要性
    total_params = sum(s['n_params'] for s in layer_stats.values())
    total_importance = sum(
        s['n_params'] * s['mean_importance'] for s in layer_stats.values()
    )
    global_mean = total_importance / total_params if total_params > 0 else 1.0

    # 计算每层的初始剪枝比例（基于重要性反比）
    # p_layer = base_ratio * (global_mean / layer_mean)
    ratios = {}
    for name, stats in layer_stats.items():
        layer_mean = stats['mean_importance']
        if layer_mean < 1e-10:
            layer_mean = 1e-10
        # 重要性越低，剪枝比例越高
        ratio = target_sparsity * (global_mean / layer_mean)
        ratio = max(0.0, min(1.0, ratio))
        ratios[name] = ratio

    # 调整比例使总剪枝参数数接近目标
    # 计算当前总剪枝参数数
    total_pruned = sum(
        layer_stats[name]['n_params'] * ratio
        for name, ratio in ratios.items()
    )
    target_pruned = total_params * target_sparsity

    # 缩放比例
    if total_pruned > 0:
        scale = target_pruned / total_pruned
        ratios = {
            name: max(0.0, min(1.0, ratio * scale))
            for name, ratio in ratios.items()
        }

    return ratios


def prune_by_layer_distribution(
    weights: Dict[str, torch.Tensor],
    importance_scores: Dict[str, torch.Tensor],
    layer_dist_params: Dict[str, Dict],
    target_sparsity: float
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, float]]:
    """
    基于分布参数的分层剪枝。

    使用 LayerPruningOptimizer 计算最优的分层剪枝比例，
    然后应用分层剪枝。

    参数:
        weights: 模型权重 {name: tensor}
        importance_scores: 重要性得分 {name: tensor}
        layer_dist_params: 每层的分布参数 {name: {N, dist_type, k, beta, ...}}
        target_sparsity: 目标全局稀疏度

    返回:
        (pruned_weights, masks, layer_ratios) 元组
    """
    if not weights:
        return {}, {}, {}

    # 计算总参数数和目标损失预算
    total_params = sum(p['N'] for p in layer_dist_params.values())

    # 使用简化的方法：基于分布参数计算每层的剪枝比例
    # 计算每层的平均重要性（使用分布参数估计）
    layer_stats = {}
    for name, params in layer_dist_params.items():
        n_params = params['N']
        # 使用 beta (尺度参数) 作为平均重要性的估计
        mean_importance = params.get('beta', 0.1)
        layer_stats[name] = {
            'n_params': n_params,
            'mean_importance': mean_importance
        }

    # 计算每层的剪枝比例
    layer_ratios = compute_layer_sparsities_from_global(layer_stats, target_sparsity)

    # 应用分层剪枝
    pruned_weights, masks = prune_by_layer_sparsity(
        weights, importance_scores, layer_ratios
    )

    return pruned_weights, masks, layer_ratios
