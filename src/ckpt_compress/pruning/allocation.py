"""统一剪枝率分配策略。"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from abc import ABC, abstractmethod
from scipy import stats
from scipy.optimize import bisect


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
        return {name: global_prune_ratio for name in scores}


class GammaAdaptiveAllocation(AllocationStrategy):
    """Gamma-adaptive: 拟合 Gamma 分布，二分法求全局阈值。
    
    论文核心方法：
    1. 对每层 damage score 拟合 Gamma 分布（矩估计）
    2. 二分法求解全局阈值 c*
    3. 每层剪枝率 p_l = F_l(c*)
    """
    
    @property
    def name(self) -> str:
        return "gamma-adaptive"

    @staticmethod
    def _estimate_quantile_from_scores(
        scores: Dict[str, torch.Tensor],
        q: float,
        max_samples: int = 2_000_000,
    ) -> float:
        """在大张量场景下稳定估计分位数，避免对超大拼接张量直接调用 quantile。"""
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be in [0, 1], got {q}")

        all_chunks: List[torch.Tensor] = []
        total_elems = 0
        for s in scores.values():
            flat = s.detach().flatten().float().cpu()
            all_chunks.append(flat)
            total_elems += flat.numel()

        if total_elems == 0:
            return 0.0

        if total_elems <= max_samples:
            merged = torch.cat(all_chunks)
            return float(torch.quantile(merged, q).item())

        per_layer_budget = max(1, max_samples // max(1, len(all_chunks)))
        sampled: List[torch.Tensor] = []
        for flat in all_chunks:
            n = flat.numel()
            if n <= per_layer_budget:
                sampled.append(flat)
            else:
                idx = torch.randperm(n)[:per_layer_budget]
                sampled.append(flat[idx])

        merged = torch.cat(sampled)
        return float(torch.quantile(merged, q).item())
    
    def allocate(self, scores, global_prune_ratio):
        # 1. 对每层拟合 Gamma 分布
        layer_info = []
        for name, score in scores.items():
            data = score.flatten().float()
            data_pos = data[data > 0]
            if len(data_pos) < 10:
                layer_info.append({'name': name, 'n': score.numel(), 'k': None, 'theta': None})
                continue
            mean = data_pos.mean().item()
            var = data_pos.var().item()
            if var < 1e-12 or mean < 1e-12:
                layer_info.append({'name': name, 'n': score.numel(), 'k': None, 'theta': None})
                continue
            k = mean ** 2 / var
            theta = var / mean
            layer_info.append({'name': name, 'n': score.numel(), 'k': k, 'theta': theta})
        
        # 2. 二分法求全局阈值
        N = sum(info['n'] for info in layer_info)
        
        def objective(c):
            total = 0
            for info in layer_info:
                if info['k'] is None:
                    total += info['n'] * global_prune_ratio  # fallback
                else:
                    cdf_val = stats.gamma.cdf(c, info['k'], scale=info['theta'])
                    total += info['n'] * cdf_val
            return total / N - global_prune_ratio
        
        # 搜索范围
        c_min = self._estimate_quantile_from_scores(scores, 0.001)
        c_max = self._estimate_quantile_from_scores(scores, 0.999)
        if c_min <= 0:
            c_min = 1e-10
        
        try:
            c_star = bisect(objective, c_min, c_max, xtol=1e-10, maxiter=200)
        except ValueError:
            # fallback to uniform
            return {name: global_prune_ratio for name in scores}
        
        # 3. 计算每层剪枝率
        result = {}
        for info in layer_info:
            if info['k'] is None:
                result[info['name']] = global_prune_ratio
            else:
                result[info['name']] = float(stats.gamma.cdf(c_star, info['k'], scale=info['theta']))
        
        return result


# 注册表
ALLOCATION_REGISTRY = {
    "uniform": UniformAllocation,
    "gamma-adaptive": GammaAdaptiveAllocation,
}

def get_allocation_strategy(name: str, **kwargs) -> AllocationStrategy:
    if name not in ALLOCATION_REGISTRY:
        raise ValueError(f"Unknown allocation: {name}. Available: {list(ALLOCATION_REGISTRY.keys())}")
    return ALLOCATION_REGISTRY[name](**kwargs)

def list_allocation_strategies():
    return list(ALLOCATION_REGISTRY.keys())
