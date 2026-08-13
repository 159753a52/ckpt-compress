"""统一剪枝率分配策略。"""

import math
from abc import ABC, abstractmethod
from typing import Dict, Type

import numpy as np
import torch
from scipy import stats
from scipy.optimize import bisect

from .validation import validate_unit_interval

_GLOBAL_SORT_MAX_PARAMS = 10_000_000


def _exact_bounded_counts(
    proposed: Dict[str, float],
    sizes: Dict[str, int],
    capacities: Dict[str, int],
    target: int,
) -> Dict[str, int]:
    """Project proposed real counts onto one deterministic exact integer budget."""
    if set(proposed) != set(sizes) or set(capacities) != set(sizes):
        raise ValueError("Proposed counts, sizes, and capacities must share keys")
    if target < 0 or target > sum(capacities.values()):
        raise ValueError("Layer capacities cannot meet the requested global pruning target")
    clipped = {
        name: min(float(capacities[name]), max(0.0, float(proposed[name]))) for name in sizes
    }
    if not all(np.isfinite(value) for value in clipped.values()):
        raise ValueError("Proposed pruning counts must be finite")
    counts = {name: int(np.floor(value)) for name, value in clipped.items()}
    difference = target - sum(counts.values())
    if difference > 0:
        order = sorted(
            sizes,
            key=lambda name: (clipped[name] - counts[name], -list(sizes).index(name)),
            reverse=True,
        )
        while difference:
            progressed = False
            for name in order:
                if counts[name] >= capacities[name]:
                    continue
                counts[name] += 1
                difference -= 1
                progressed = True
                if difference == 0:
                    break
            if not progressed:
                raise RuntimeError("Could not satisfy the exact pruning budget")
    elif difference < 0:
        order = sorted(
            sizes,
            key=lambda name: (clipped[name] - counts[name], list(sizes).index(name)),
        )
        while difference:
            progressed = False
            for name in order:
                if counts[name] == 0:
                    continue
                counts[name] -= 1
                difference += 1
                progressed = True
                if difference == 0:
                    break
            if not progressed:
                raise RuntimeError("Could not reduce counts to the exact pruning budget")
    return counts


def _ratios_from_counts(counts: Dict[str, int], sizes: Dict[str, int]) -> Dict[str, float]:
    ratios = {}
    for name, size in sizes.items():
        count = counts[name]
        if size == 0 or count == 0:
            ratios[name] = 0.0
            continue
        if count == size:
            ratios[name] = 1.0
            continue

        ratio = count / size
        # Downstream callers recover counts with ``int(size * ratio)``.
        # Correct the occasional one-ULP underflow from division/multiplication.
        while int(size * ratio) < count:
            ratio = math.nextafter(ratio, 1.0)
        if int(size * ratio) != count:
            raise RuntimeError("Could not encode the exact pruning count as a ratio")
        ratios[name] = ratio
    return ratios


def _flatten_finite_scores(
    scores: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Detach, flatten, and validate score tensors once per allocation call."""
    flattened = {}
    for name, score in scores.items():
        flat = score.detach().flatten().float().cpu()
        if not torch.isfinite(flat).all():
            raise ValueError(f"Scores for {name!r} must contain finite values")
        flattened[name] = flat
    return flattened


def _exact_counts_from_threshold(
    flat_layers: Dict[str, torch.Tensor],
    threshold: torch.Tensor,
    prune_count: int,
) -> Dict[str, int]:
    """Distribute values below a threshold and then resolve ties deterministically."""
    counts = {name: 0 for name in flat_layers}
    tie_capacity = {}
    assigned = 0
    for name, values in flat_layers.items():
        below = int((values < threshold).sum().item())
        at_or_below = int((values <= threshold).sum().item())
        counts[name] = below
        tie_capacity[name] = at_or_below - below
        assigned += below

    remaining = prune_count - assigned
    if remaining < 0:
        raise RuntimeError("Global pruning threshold exceeds the requested budget")
    for name in flat_layers:
        take = min(remaining, tie_capacity[name])
        counts[name] += take
        remaining -= take
    if remaining:
        raise RuntimeError("Unable to satisfy the exact global pruning budget")
    return counts


class AllocationStrategy(ABC):
    """剪枝率分配策略的抽象基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def allocate(
        self,
        scores: Dict[str, torch.Tensor],
        global_prune_ratio: float,
    ) -> Dict[str, float]:
        """计算每层的剪枝率。

        返回: {layer_name: prune_ratio} 字典
        """
        pass


class UniformAllocation(AllocationStrategy):
    """Uniform: 所有层使用相同的剪枝率。"""

    @property
    def name(self) -> str:
        return "uniform"

    def allocate(self, scores, global_prune_ratio):
        ratio = validate_unit_interval("global_prune_ratio", global_prune_ratio)
        return {name: ratio for name in scores}


class GlobalTopKAllocation(AllocationStrategy):
    """Global Top-K: 精确全局排序，计算每层应保留的 K 值。

    作为 Gamma-adaptive 的上限对比：
    1. 将所有层的 score 合并排序
    2. 取全局 top-K 保留
    3. 计算每层应保留的数量，反推剪枝率

    优点：精确控制全局稀疏度
    小规模使用稳定全局排序；大规模使用线性 kth-value 阈值选择。
    """

    @property
    def name(self) -> str:
        return "global-topk"

    def allocate(
        self,
        scores: Dict[str, torch.Tensor],
        global_prune_ratio: float,
    ) -> Dict[str, float]:
        """计算每层的剪枝率。

        参数:
            scores: 重要性得分享典
            global_prune_ratio: 全局剪枝比例

        返回:
            {layer_name: prune_ratio} 字典
        """
        global_prune_ratio = validate_unit_interval("global_prune_ratio", global_prune_ratio)

        if not scores:
            return {}

        # 1. 计算每层参数数量
        layer_sizes = {name: score.numel() for name, score in scores.items()}
        total_params = sum(layer_sizes.values())
        if total_params == 0:
            return {name: 0.0 for name in scores}
        if global_prune_ratio == 0.0:
            return {name: 0.0 for name in scores}
        if global_prune_ratio == 1.0:
            return {name: 1.0 if layer_sizes[name] else 0.0 for name in scores}

        prune_count = int(total_params * global_prune_ratio)
        flat_scores = _flatten_finite_scores(scores)

        # 2. 分数已统一转到 CPU，供后续精确选择或稳定排序使用。
        all_scores = list(flat_scores.values())

        # 小规模路径保留全局排序的稳定顺序，并将精确的剪枝计数回填到各层。
        # 这避免了分数并列时仅依赖 quantile 阈值造成的预算漂移。
        if total_params <= _GLOBAL_SORT_MAX_PARAMS:
            merged = torch.cat(all_scores)
            order = torch.argsort(merged, stable=True)
            boundaries = torch.tensor(
                np.cumsum(list(layer_sizes.values())[:-1]),
                dtype=torch.long,
            )
            selected_layers = torch.bucketize(order[:prune_count], boundaries, right=True)
            pruned_counts = torch.bincount(
                selected_layers,
                minlength=len(layer_sizes),
            )
            counts = {
                name: int(pruned_counts[index].item()) for index, name in enumerate(layer_sizes)
            }
            return _ratios_from_counts(counts, layer_sizes)

        # 大规模路径用全局 kth-value 找阈值，再逐层统计并列值。
        # 这避免了全局排序索引和逐层 O(n log n) 排序，同时保持精确预算。
        if prune_count == 0:
            return {name: 0.0 for name in scores}
        merged = torch.cat(all_scores)
        threshold = merged.kthvalue(prune_count).values
        pruned_counts = _exact_counts_from_threshold(
            flat_scores,
            threshold,
            prune_count,
        )
        del merged
        del all_scores
        return _ratios_from_counts(pruned_counts, layer_sizes)


class GammaAdaptiveAllocation(AllocationStrategy):
    """Distribution-aware adaptive allocation via bisection on empirical CDF.

    论文核心方法：
    1. 预排序每层 damage scores
    2. 二分法搜索全局阈值 c*（每步用 binary search 统计 ≤c* 的参数数）
    3. 每层剪枝率 = (scores ≤ c*) / n_layer

    等价于 Global-TopK 但通过 bisection + searchsorted 实现，
    当层数 L << 总参数 N 时避免全局排序。
    """

    @property
    def name(self) -> str:
        return "gamma-adaptive"

    def allocate(
        self,
        scores: Dict[str, torch.Tensor],
        global_prune_ratio: float,
    ) -> Dict[str, float]:
        global_prune_ratio = validate_unit_interval("global_prune_ratio", global_prune_ratio)

        flat_scores = _flatten_finite_scores(scores)
        sizes = {name: flat.numel() for name, flat in flat_scores.items()}
        total = sum(sizes.values())
        if total == 0:
            return {name: 0.0 for name in scores}
        target = int(total * global_prune_ratio)
        if target == 0:
            return {name: 0.0 for name in scores}
        merged = torch.cat(list(flat_scores.values()))
        threshold = merged.kthvalue(target).values
        counts = _exact_counts_from_threshold(flat_scores, threshold, target)
        return _ratios_from_counts(counts, sizes)


class WeibullAdaptiveAllocation(AllocationStrategy):
    """Weibull-adaptive: 拟合 Weibull 分布做 CDF bisection。

    Weibull 通常比 Gamma 拟合 damage score 更好（更低 KS-D）。

    参数:
        max_layer_ratio: 每层最大剪枝率上限（防止 FT 中极端剪枝）
    """

    def __init__(self, max_layer_ratio: float = 0.5):
        self.max_layer_ratio = validate_unit_interval("max_layer_ratio", max_layer_ratio)

    @property
    def name(self) -> str:
        return "weibull-adaptive"

    def allocate(
        self,
        scores: Dict[str, torch.Tensor],
        global_prune_ratio: float,
    ) -> Dict[str, float]:
        global_prune_ratio = validate_unit_interval("global_prune_ratio", global_prune_ratio)
        MAX_FIT_SAMPLES = 50000

        # 1. 对每层拟合 Weibull
        flat_scores = _flatten_finite_scores(scores)
        layer_info = []
        for name, data in flat_scores.items():
            data_pos = data[data > 0].cpu().numpy()
            if len(data_pos) < 10:
                layer_info.append({"name": name, "n": data.numel(), "params": None})
                continue
            if len(data_pos) > MAX_FIT_SAMPLES:
                rng = np.random.RandomState(42)
                data_pos = rng.choice(data_pos, MAX_FIT_SAMPLES, replace=False)
            try:
                params = stats.weibull_min.fit(data_pos, floc=0)
                layer_info.append({"name": name, "n": data.numel(), "params": params})
            except Exception:
                layer_info.append({"name": name, "n": data.numel(), "params": None})

        N = sum(info["n"] for info in layer_info)
        if N == 0:
            return {name: 0.0 for name in scores}
        target = int(N * global_prune_ratio)
        sizes = {str(info["name"]): int(info["n"]) for info in layer_info}
        capacities = {
            name: min(size, int(np.floor(self.max_layer_ratio * size)))
            for name, size in sizes.items()
        }
        if target > sum(capacities.values()):
            raise ValueError("max_layer_ratio cannot meet global_prune_ratio")

        # 2. Weibull CDF bisection
        def objective(c: float) -> float:
            total = 0
            for info in layer_info:
                if info["params"] is None:
                    total += info["n"] * global_prune_ratio
                else:
                    ratio = stats.weibull_min.cdf(c, *info["params"])
                    ratio = min(ratio, self.max_layer_ratio)
                    total += info["n"] * ratio
            return float(total / N - global_prune_ratio)

        # 搜索范围（子采样估计 99.9% 分位数，避免大张量 quantile 溢出）
        generator = torch.Generator(device="cpu").manual_seed(42)
        sampled_for_quantile = []
        for flat in flat_scores.values():
            if flat.numel() == 0:
                continue
            if flat.numel() > 100_000:
                idx = torch.randperm(flat.numel(), generator=generator)[:100_000]
                sampled_for_quantile.append(flat[idx])
            else:
                sampled_for_quantile.append(flat)
        c_max = torch.cat(sampled_for_quantile).quantile(0.999).item()
        if c_max <= 1e-10:
            c_max = 1.0

        proposed: Dict[str, float]
        try:
            c_star = bisect(objective, 0.0, c_max, xtol=1e-10, maxiter=200)
        except ValueError:
            proposed = {
                name: min(global_prune_ratio * size, capacities[name])
                for name, size in sizes.items()
            }
        else:
            proposed = {}
            for info in layer_info:
                name = str(info["name"])
                if info["n"] == 0:
                    proposed[name] = 0.0
                    continue
                flat = flat_scores[name]
                proposed[name] = min(
                    float(capacities[name]),
                    float((flat <= c_star).sum().item()),
                )

        counts = _exact_bounded_counts(proposed, sizes, capacities, target)
        return _ratios_from_counts(counts, sizes)


# 注册表
ALLOCATION_REGISTRY: Dict[str, Type[AllocationStrategy]] = {
    "uniform": UniformAllocation,
    "gamma-adaptive": GammaAdaptiveAllocation,
    "global-topk": GlobalTopKAllocation,
    "weibull-adaptive": WeibullAdaptiveAllocation,
}


def get_allocation_strategy(name: str, **kwargs) -> AllocationStrategy:
    if name not in ALLOCATION_REGISTRY:
        raise ValueError(
            f"Unknown allocation: {name}. Available: {list(ALLOCATION_REGISTRY.keys())}"
        )
    strategy = ALLOCATION_REGISTRY[name](**kwargs)
    if not isinstance(strategy, AllocationStrategy):
        raise TypeError(f"Registered allocation {name!r} did not create an AllocationStrategy")
    return strategy


def list_allocation_strategies():
    return list(ALLOCATION_REGISTRY.keys())
