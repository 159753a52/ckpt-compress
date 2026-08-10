"""统一剪枝率分配策略。"""

import torch
import numpy as np
from typing import Dict
from abc import ABC, abstractmethod
from scipy import stats
from scipy.optimize import bisect

from .validation import validate_unit_interval


_GLOBAL_SORT_MAX_PARAMS = 10_000_000


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
        global_prune_ratio = validate_unit_interval(
            "global_prune_ratio", global_prune_ratio
        )

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
            return {
                name: 1.0 if layer_sizes[name] else 0.0
                for name in scores
            }

        prune_count = int(total_params * global_prune_ratio)
        
        # 2. 将分数转到 CPU，供后续精确选择或稳定排序使用。
        all_scores = []
        for name, score in scores.items():
            flat = score.flatten().float().cpu()
            if not torch.isfinite(flat).all():
                raise ValueError(f"Scores for {name!r} must contain finite values")
            all_scores.append(flat)

        # 小规模路径保留全局排序的稳定顺序，并将精确的剪枝计数回填到各层。
        # 这避免了分数并列时仅依赖 quantile 阈值造成的预算漂移。
        if total_params <= _GLOBAL_SORT_MAX_PARAMS:
            merged = torch.cat(all_scores)
            order = torch.argsort(merged, stable=True)
            boundaries = torch.tensor(
                np.cumsum(list(layer_sizes.values())[:-1]),
                dtype=torch.long,
            )
            selected_layers = torch.bucketize(
                order[:prune_count], boundaries, right=True
            )
            pruned_counts = torch.bincount(
                selected_layers,
                minlength=len(layer_sizes),
            )
            return {
                name: (
                    float(pruned_counts[index].item()) / layer_sizes[name]
                    if layer_sizes[name] > 0
                    else 0.0
                )
                for index, name in enumerate(layer_sizes)
            }

        # 大规模路径用全局 kth-value 找阈值，再逐层统计并列值。
        # 这避免了全局排序索引和逐层 O(n log n) 排序，同时保持精确预算。
        if prune_count == 0:
            return {name: 0.0 for name in scores}
        merged = torch.cat(all_scores)
        threshold = merged.kthvalue(prune_count).values
        pruned_counts = _exact_counts_from_threshold(
            dict(zip(layer_sizes, all_scores)),
            threshold,
            prune_count,
        )
        del merged
        del all_scores
        return {
            name: (
                pruned_counts[name] / layer_sizes[name]
                if layer_sizes[name] > 0
                else 0.0
            )
            for name in layer_sizes
        }


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
        global_prune_ratio = validate_unit_interval(
            "global_prune_ratio", global_prune_ratio
        )

        # 1. 预排序每层 scores
        sorted_layers = {}
        sizes = {}
        for name, score in scores.items():
            flat = score.flatten().float().cpu()
            sorted_layers[name] = flat.sort().values
            sizes[name] = flat.numel()
        
        N = sum(sizes.values())
        if N == 0:
            return {name: 0.0 for name in scores}
        
        # 2. 用经验 CDF 做 bisection
        def objective(c: float) -> float:
            c_tensor = torch.tensor(c)
            total_below = sum(
                torch.searchsorted(sorted_layers[name], c_tensor).item()
                for name in sorted_layers
            )
            return total_below / N - global_prune_ratio
        
        c_min = 0.0
        c_max = max(
            s[-1].item() for s in sorted_layers.values() if s.numel() > 0
        )
        if c_max <= 1e-10:
            c_max = 1.0
        
        try:
            c_star = bisect(objective, c_min, c_max, xtol=1e-10, maxiter=200)
        except ValueError:
            return {name: global_prune_ratio for name in scores}
        
        # 3. 每层剪枝率
        c_tensor = torch.tensor(c_star)
        result = {}
        for name in scores:
            if sizes[name] == 0:
                result[name] = 0.0
                continue
            idx = torch.searchsorted(sorted_layers[name], c_tensor).item()
            result[name] = max(0.0, min(1.0, idx / sizes[name]))
        
        return result


class WeibullAdaptiveAllocation(AllocationStrategy):
    """Weibull-adaptive: 拟合 Weibull 分布做 CDF bisection。
    
    Weibull 通常比 Gamma 拟合 damage score 更好（更低 KS-D）。
    
    参数:
        max_layer_ratio: 每层最大剪枝率上限（防止 FT 中极端剪枝）
    """
    
    def __init__(self, max_layer_ratio: float = 0.5):
        self.max_layer_ratio = validate_unit_interval(
            "max_layer_ratio", max_layer_ratio
        )
    
    @property
    def name(self) -> str:
        return "weibull-adaptive"
    
    def allocate(
        self,
        scores: Dict[str, torch.Tensor],
        global_prune_ratio: float,
    ) -> Dict[str, float]:
        global_prune_ratio = validate_unit_interval(
            "global_prune_ratio", global_prune_ratio
        )
        MAX_FIT_SAMPLES = 50000
        
        # 1. 对每层拟合 Weibull
        layer_info = []
        for name, score in scores.items():
            data = score.flatten().float()
            data_pos = data[data > 0].cpu().numpy()
            if len(data_pos) < 10:
                layer_info.append({'name': name, 'n': score.numel(), 'params': None})
                continue
            if len(data_pos) > MAX_FIT_SAMPLES:
                rng = np.random.RandomState(42)
                data_pos = rng.choice(data_pos, MAX_FIT_SAMPLES, replace=False)
            try:
                params = stats.weibull_min.fit(data_pos, floc=0)
                layer_info.append({'name': name, 'n': score.numel(), 'params': params})
            except Exception:
                layer_info.append({'name': name, 'n': score.numel(), 'params': None})
        
        N = sum(info['n'] for info in layer_info)
        if N == 0:
            return {name: 0.0 for name in scores}
        
        # 2. Weibull CDF bisection
        def objective(c: float) -> float:
            total = 0
            for info in layer_info:
                if info['params'] is None:
                    total += info['n'] * global_prune_ratio
                else:
                    ratio = stats.weibull_min.cdf(c, *info['params'])
                    ratio = min(ratio, self.max_layer_ratio)
                    total += info['n'] * ratio
            return total / N - global_prune_ratio
        
        # 搜索范围（子采样估计 99.9% 分位数，避免大张量 quantile 溢出）
        generator = torch.Generator(device='cpu').manual_seed(42)
        sampled_for_quantile = []
        for score in scores.values():
            flat = score.flatten().float().cpu()
            if flat.numel() == 0:
                continue
            if flat.numel() > 100_000:
                idx = torch.randperm(
                    flat.numel(), generator=generator
                )[:100_000]
                sampled_for_quantile.append(flat[idx])
            else:
                sampled_for_quantile.append(flat)
        c_max = torch.cat(sampled_for_quantile).quantile(0.999).item()
        if c_max <= 1e-10:
            c_max = 1.0
        
        try:
            c_star = bisect(objective, 0.0, c_max, xtol=1e-10, maxiter=200)
        except ValueError:
            return {name: global_prune_ratio for name in scores}
        
        # 3. 最终层率用经验计数 + per-layer cap
        result = {}
        for info in layer_info:
            name = info['name']
            if info['n'] == 0:
                result[name] = 0.0
                continue
            flat = scores[name].flatten().float()
            below = (flat <= c_star).sum().item()
            ratio = max(0.0, min(self.max_layer_ratio, below / info['n']))
            result[name] = ratio
        
        return result


# 注册表
ALLOCATION_REGISTRY = {
    "uniform": UniformAllocation,
    "gamma-adaptive": GammaAdaptiveAllocation,
    "global-topk": GlobalTopKAllocation,
    "weibull-adaptive": WeibullAdaptiveAllocation,
}

def get_allocation_strategy(name: str, **kwargs) -> AllocationStrategy:
    if name not in ALLOCATION_REGISTRY:
        raise ValueError(f"Unknown allocation: {name}. Available: {list(ALLOCATION_REGISTRY.keys())}")
    return ALLOCATION_REGISTRY[name](**kwargs)

def list_allocation_strategies():
    return list(ALLOCATION_REGISTRY.keys())
