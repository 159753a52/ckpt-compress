"""统一重要性得分计算接口。"""

import torch
from typing import Dict, Optional
from abc import ABC, abstractmethod


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
        """是否需要 Hessian（或 Adam exp_avg_sq）信息。"""
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
        exp_avg_sq: Optional[Dict[str, torch.Tensor]] = None,
        reference_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算每个参数的重要性得分。返回值越大表示越重要。"""
        pass


class MagnitudeScorer(ImportanceScorer):
    """Magnitude-based: |w_i|"""
    
    @property
    def name(self) -> str:
        return "magnitude"
    
    def score(self, weights, gradients=None, exp_avg_sq=None, reference_weights=None):
        return {name: torch.abs(w) for name, w in weights.items()}


class FirstOrderScorer(ImportanceScorer):
    """Inshrinkerator-style 一阶: |g_i · w_i|"""
    
    @property
    def name(self) -> str:
        return "first-order"
    
    @property
    def requires_gradients(self) -> bool:
        return True
    
    def score(self, weights, gradients=None, exp_avg_sq=None, reference_weights=None):
        scores = {}
        for name, w in weights.items():
            g = gradients.get(name, torch.zeros_like(w))
            scores[name] = torch.abs(g * w)
        return scores


class SecondOrderScorer(ImportanceScorer):
    """Ours: |-g_i·w_i + 0.5·h_i·w_i²|
    
    使用 Adam 的 exp_avg_sq 作为 Hessian 对角线近似。
    """
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
    
    @property
    def name(self) -> str:
        return "second-order"
    
    @property
    def requires_gradients(self) -> bool:
        return True
    
    @property
    def requires_hessian(self) -> bool:
        return True
    
    def score(self, weights, gradients=None, exp_avg_sq=None, reference_weights=None):
        scores = {}
        for name, w in weights.items():
            g = gradients.get(name, torch.zeros_like(w))
            h = exp_avg_sq.get(name, torch.zeros_like(w))
            # s_i = |-g_i * w_i + alpha * h_i * w_i^2|
            first_order = -g * w
            second_order = self.alpha * h * w ** 2
            scores[name] = torch.abs(first_order + second_order)
        return scores


class ResidualMagnitudeScorer(ImportanceScorer):
    """ExCP-style: |W_t - W_ref| 残差 magnitude。"""
    
    @property
    def name(self) -> str:
        return "residual-magnitude"
    
    @property
    def requires_reference(self) -> bool:
        return True
    
    def score(self, weights, gradients=None, exp_avg_sq=None, reference_weights=None):
        scores = {}
        for name, w in weights.items():
            if reference_weights and name in reference_weights:
                scores[name] = torch.abs(w - reference_weights[name])
            else:
                scores[name] = torch.abs(w)
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
    return IMPORTANCE_REGISTRY[name](**kwargs)

def list_importance_methods():
    """列出所有已注册的重要性方法。"""
    return list(IMPORTANCE_REGISTRY.keys())

# 注册内置方法
register_importance(MagnitudeScorer)
register_importance(FirstOrderScorer)
# SecondOrderScorer 需要参数，手动注册
IMPORTANCE_REGISTRY["second-order"] = SecondOrderScorer
IMPORTANCE_REGISTRY["residual-magnitude"] = ResidualMagnitudeScorer
