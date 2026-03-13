"""统一剪枝执行器。"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List

from .importance import ImportanceScorer, get_importance_scorer
from .allocation import AllocationStrategy, get_allocation_strategy


def filter_prunable_params(
    weights: Dict[str, torch.Tensor],
    exclude_patterns: List[str] = None,
) -> Dict[str, torch.Tensor]:
    """过滤可剪枝参数（排除 embedding、bias、layernorm 等）。
    
    默认排除: embedding 层、bias 参数
    """
    if exclude_patterns is None:
        exclude_patterns = ['embed', 'wte', 'wpe', 'bias', 'ln_', 'LayerNorm', 'layernorm']
    
    filtered = {}
    for name, w in weights.items():
        if any(pat in name for pat in exclude_patterns):
            continue
        if w.dim() < 2:  # 跳过 1D 参数（通常是 bias 或 norm）
            continue
        filtered[name] = w
    return filtered


def apply_pruning(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    layer_ratios: Dict[str, float],
    device: str = 'cpu',
) -> Tuple[nn.Module, Dict[str, torch.Tensor], float]:
    """对模型应用剪枝。
    
    参数:
        model: PyTorch 模型
        scores: 重要性得分
        layer_ratios: 每层剪枝率
        device: 设备
    
    返回:
        (model, masks, actual_global_ratio)
    """
    masks = {}
    total_pruned = 0
    total_params = 0
    
    model_params = dict(model.named_parameters())
    
    for name, score in scores.items():
        if name not in model_params or name not in layer_ratios:
            continue
        
        ratio = layer_ratios[name]
        flat_score = score.flatten()
        k = int(len(flat_score) * ratio)
        
        if k == 0:
            masks[name] = torch.ones_like(score)
            continue
        
        threshold = torch.kthvalue(flat_score, k).values
        mask = (score > threshold).float()
        masks[name] = mask
        
        # 应用掩码
        param = model_params[name]
        param.data.mul_(mask.to(device))
        
        total_pruned += (mask == 0).sum().item()
        total_params += mask.numel()
    
    actual_ratio = total_pruned / total_params if total_params > 0 else 0
    return model, masks, actual_ratio


class Pruner:
    """统一剪枝器，组合 importance scorer + allocation strategy。
    
    用法:
        pruner = Pruner(importance='second-order', allocation='gamma-adaptive', alpha=0.5)
        scores = pruner.compute_scores(weights, gradients, exp_avg_sq)
        layer_ratios = pruner.compute_layer_ratios(scores, global_ratio=0.5)
        model, masks, actual_ratio = pruner.prune(model, scores, layer_ratios, device)
    """
    
    def __init__(
        self,
        importance: str = 'second-order',
        allocation: str = 'gamma-adaptive',
        **kwargs,
    ):
        self.scorer = get_importance_scorer(importance, **{k: v for k, v in kwargs.items() if k == 'alpha'})
        self.allocator = get_allocation_strategy(allocation)
        self.importance_name = importance
        self.allocation_name = allocation
    
    @property
    def name(self) -> str:
        return f"{self.importance_name}+{self.allocation_name}"
    
    def compute_scores(
        self,
        weights: Dict[str, torch.Tensor],
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        exp_avg_sq: Optional[Dict[str, torch.Tensor]] = None,
        reference_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.scorer.score(weights, gradients, exp_avg_sq, reference_weights)
    
    def compute_layer_ratios(
        self,
        scores: Dict[str, torch.Tensor],
        global_ratio: float,
    ) -> Dict[str, float]:
        return self.allocator.allocate(scores, global_ratio)
    
    def prune(
        self,
        model: nn.Module,
        scores: Dict[str, torch.Tensor],
        layer_ratios: Dict[str, float],
        device: str = 'cpu',
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor], float]:
        return apply_pruning(model, scores, layer_ratios, device)
