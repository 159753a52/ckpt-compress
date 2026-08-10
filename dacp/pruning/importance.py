"""统一重要性得分计算接口。"""

import torch
import math
from typing import Dict, Optional
from abc import ABC, abstractmethod


def _validate_unit_interval(name: str, value: float) -> float:
    """Validate a scalar hyperparameter constrained to the unit interval."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number in [0, 1], got {value}")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite number in [0, 1], got {value}"
        ) from exc
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1], got {value}")
    return normalized


class ImportanceScorer(ABC):
    """重要性得分计算的抽象基类。"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """方法名称。"""
        pass
    
    @property
    def requires_gradients(self) -> bool:
        """是否需要梯度信息。"""
        return False
    
    @property
    def requires_hessian(self) -> bool:
        """是否需要 Hessian 信息。"""
        return False
    
    @property
    def requires_reference(self) -> bool:
        """是否需要参考权重（如 ExCP 的前一检查点）。"""
        return False
    
    @abstractmethod
    def score(
        self,
        weights: Dict[str, torch.Tensor],
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        reference_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算每个参数的重要性得分。返回值越大表示越重要。"""
        pass


class MagnitudeScorer(ImportanceScorer):
    """Magnitude-based: |w_i|"""
    
    @property
    def name(self) -> str:
        return "magnitude"
    
    def score(self, weights, gradients=None, reference_weights=None):
        return {name: torch.abs(w) for name, w in weights.items()}


class FirstOrderScorer(ImportanceScorer):
    """Inshrinkerator-style 一阶: |g_i · w_i|"""
    
    @property
    def name(self) -> str:
        return "first-order"
    
    @property
    def requires_gradients(self) -> bool:
        return True
    
    def score(self, weights, gradients=None, reference_weights=None):
        scores = {}
        for name, w in weights.items():
            g = gradients.get(name, torch.zeros_like(w))
            scores[name] = torch.abs(g * w)
        return scores



class ResidualMagnitudeScorer(ImportanceScorer):
    """ExCP-style: |W_t - W_ref| 残差 magnitude。"""
    
    @property
    def name(self) -> str:
        return "residual-magnitude"
    
    @property
    def requires_reference(self) -> bool:
        return True
    
    def score(self, weights, gradients=None, reference_weights=None):
        scores = {}
        for name, w in weights.items():
            if reference_weights and name in reference_weights:
                scores[name] = torch.abs(w - reference_weights[name])
            else:
                scores[name] = torch.abs(w)
        return scores


class RandomScorer(ImportanceScorer):
    """随机重要性得分——控制组基线。"""

    @property
    def name(self) -> str:
        return 'random'

    def score(self, weights, gradients=None, reference_weights=None):
        scores = {}
        for name, w in weights.items():
            scores[name] = torch.rand_like(w)
        return scores


# ============================================================
# 注册表
# ============================================================

IMPORTANCE_REGISTRY = {}

def register_importance(cls):
    """装饰器：注册重要性得分方法。"""
    instance = cls() if not isinstance(cls, ImportanceScorer) else cls
    IMPORTANCE_REGISTRY[instance.name] = cls
    return cls

def get_importance_scorer(name: str, **kwargs) -> ImportanceScorer:
    """根据名称获取重要性得分计算器。"""
    if name not in IMPORTANCE_REGISTRY:
        raise ValueError(f"Unknown importance method: {name}. Available: {list(IMPORTANCE_REGISTRY.keys())}")
    registered = IMPORTANCE_REGISTRY[name]
    if isinstance(registered, ImportanceScorer):
        if kwargs:
            raise TypeError(
                f"Registered scorer {name!r} is an instance and does not accept "
                f"constructor arguments: {sorted(kwargs)}"
            )
        return registered
    return registered(**kwargs)

def list_importance_methods():
    """列出所有已注册的重要性方法。"""
    return list(IMPORTANCE_REGISTRY.keys())

# 注册内置方法
register_importance(MagnitudeScorer)
register_importance(FirstOrderScorer)
register_importance(ResidualMagnitudeScorer)
register_importance(RandomScorer)


# ============================================================
# 2D 组合得分与 Protection
# ============================================================

def combine_scores_2d_with_protection(
    magnitude_scores: Dict[str, torch.Tensor],
    damage_scores: Dict[str, torch.Tensor],
    protection_ratio: float = 0.001,
    alpha: float = 0.7,
) -> Dict[str, torch.Tensor]:
    """2D 剪枝：加权组合 magnitude + damage score，附带 outlier protection。

    逻辑：
    1. 每层内，对每个参数分别计算 magnitude 和 damage 的百分位排名 ∈ [0, 1]
    2. 组合分数 = α·rank_mag + (1-α)·rank_damage
       → magnitude 为主导信号（干净），damage 为修正量
    3. Protection：任一维度排名在 top protection_ratio 的参数 → 分数设为 1.0

    参数:
        magnitude_scores: {name: |θ|} 幅值得分
        damage_scores: {name: damage} damage 得分（如 HVP-based）
        protection_ratio: 每维度保护的参数比例（默认 0.1%）
        alpha: magnitude 权重（默认 0.7），越大 magnitude 影响越强

    返回:
        组合得分 {name: tensor} ∈ [0, 1]
    """
    protection_ratio = _validate_unit_interval("protection_ratio", protection_ratio)
    alpha = _validate_unit_interval("alpha", alpha)
    combined = {}

    for name in damage_scores:
        if name not in magnitude_scores:
            continue
        if magnitude_scores[name].shape != damage_scores[name].shape:
            raise ValueError(
                f"Score shapes for {name!r} must match: "
                f"{tuple(magnitude_scores[name].shape)} != "
                f"{tuple(damage_scores[name].shape)}"
            )

        mag = magnitude_scores[name].flatten().float()
        dam = damage_scores[name].flatten().float()
        n = mag.numel()

        if n == 0:
            combined[name] = mag.clone()
            continue

        # 百分位排名 ∈ [0, (n-1)/n]
        mag_rank = torch.argsort(
            torch.argsort(mag, stable=True), stable=True
        ).float() / n
        dam_rank = torch.argsort(
            torch.argsort(dam, stable=True), stable=True
        ).float() / n

        # 加权组合：α·magnitude + (1-α)·damage
        comb = alpha * mag_rank + (1.0 - alpha) * dam_rank

        # Protection：任一维度极高 → 不可剪
        protect_threshold = 1.0 - protection_ratio
        protect_mask = (mag_rank >= protect_threshold) | (dam_rank >= protect_threshold)
        comb[protect_mask] = 1.0

        combined[name] = comb.reshape(magnitude_scores[name].shape)

    return combined


def apply_magnitude_protection(
    scores: Dict[str, torch.Tensor],
    weights: Dict[str, torch.Tensor],
    protection_ratio: float = 0.001,
) -> Dict[str, torch.Tensor]:
    """对高 magnitude 参数施加保护，防止被剪枝。

    简单方案：保留原始 damage score 的分布结构（可用于 Weibull 拟合），
    仅将 top-K% magnitude 参数的 score 提升到极高值。

    参数:
        scores: damage 得分 {name: tensor}
        weights: 模型权重 {name: tensor}（用于计算 magnitude）
        protection_ratio: 保护的参数比例（默认 0.1%）

    返回:
        修改后的得分 {name: tensor}，protected 参数得分极高
    """
    protection_ratio = _validate_unit_interval("protection_ratio", protection_ratio)
    protected = {}

    for name, score in scores.items():
        if name not in weights:
            protected[name] = score.clone()
            continue
        if score.shape != weights[name].shape:
            raise ValueError(
                f"Score and weight shapes for {name!r} must match: "
                f"{tuple(score.shape)} != {tuple(weights[name].shape)}"
            )

        s = score.clone().float()
        mag = weights[name].abs().flatten().float()
        n = mag.numel()

        if n == 0:
            protected[name] = s
            continue

        if protection_ratio == 0.0:
            protected[name] = s.reshape(score.shape)
            continue

        protect_count = min(n, max(1, math.ceil(n * protection_ratio)))
        high_mag_order = torch.argsort(mag, descending=True, stable=True)
        protected_indices = high_mag_order[:protect_count]

        # Use a finite sentinel so protection works even when all damage scores are zero.
        s.flatten()[protected_indices] = torch.finfo(s.dtype).max

        protected[name] = s.reshape(score.shape)

    return protected
