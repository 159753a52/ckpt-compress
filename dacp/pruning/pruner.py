"""统一剪枝执行器。"""

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .allocation import get_allocation_strategy
from .importance import get_importance_scorer
from .masks import exact_keep_mask
from .validation import validate_unit_interval


def exact_pruning_mask(score: torch.Tensor, prune_count: int) -> torch.Tensor:
    """Build a deterministic float mask that prunes exactly ``prune_count`` values."""
    return exact_keep_mask(score, prune_count).float()


def filter_prunable_params(
    weights: Dict[str, torch.Tensor],
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """过滤可剪枝参数（排除 embedding、bias、layernorm 等）。

    默认排除: embedding 层、bias 参数
    """
    if exclude_patterns is None:
        exclude_patterns = ["embed", "wte", "wpe", "bias", "ln_", "LayerNorm", "layernorm"]

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
    device: str = "cpu",
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
    application_masks = {}
    total_pruned = 0
    total_params = 0

    model_params = dict(model.named_parameters())
    if set(layer_ratios) != set(scores):
        missing = sorted(set(scores).difference(layer_ratios))
        extra = sorted(set(layer_ratios).difference(scores))
        raise ValueError(f"layer_ratios must match score keys: missing={missing}, extra={extra}")
    missing_parameters = sorted(set(scores).difference(model_params))
    if missing_parameters:
        raise ValueError(f"Scores reference missing model parameters: {missing_parameters}")

    # Validate every layer and prepare every mask before mutating the model.
    # A malformed later layer must not leave earlier parameters half-pruned.
    for name, score in scores.items():
        param = model_params[name]
        if score.shape != param.shape:
            raise ValueError(
                f"Score shape for {name!r} must match parameter shape: "
                f"{tuple(score.shape)} != {tuple(param.shape)}"
            )

        ratio = validate_unit_interval(f"Pruning ratio for {name!r}", layer_ratios[name])

        k = int(score.numel() * ratio)
        mask = exact_pruning_mask(score, k)
        masks[name] = mask

        # 应用掩码
        application_mask = mask.to(device=device, dtype=param.dtype)
        if application_mask.device != param.device:
            raise ValueError(
                f"Pruning device for {name!r} must match parameter device: "
                f"{application_mask.device} != {param.device}"
            )
        application_masks[name] = application_mask

        total_pruned += k
        total_params += mask.numel()

    with torch.no_grad():
        for name, application_mask in application_masks.items():
            model_params[name].mul_(application_mask)

    actual_ratio = total_pruned / total_params if total_params > 0 else 0
    return model, masks, actual_ratio


class Pruner:
    """统一剪枝器，组合 importance scorer + allocation strategy。

    用法:
        pruner = Pruner(importance='first-order', allocation='gamma-adaptive')
        scores = pruner.compute_scores(weights, gradients)
        layer_ratios = pruner.compute_layer_ratios(scores, global_ratio=0.5)
        model, masks, actual_ratio = pruner.prune(model, scores, layer_ratios, device)
    """

    def __init__(
        self,
        importance: str = "first-order",
        allocation: str = "gamma-adaptive",
        importance_kwargs: Optional[Mapping[str, Any]] = None,
        allocation_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """Create a pruner from independently configured components.

        ``importance_kwargs`` and ``allocation_kwargs`` are intentionally
        separate so a parameter intended for one component cannot be silently
        dropped or accidentally passed to the other one.
        """
        self.scorer = get_importance_scorer(
            importance,
            **dict(importance_kwargs or {}),
        )
        self.allocator = get_allocation_strategy(
            allocation,
            **dict(allocation_kwargs or {}),
        )
        self.importance_name = importance
        self.allocation_name = allocation

    @property
    def name(self) -> str:
        return f"{self.importance_name}+{self.allocation_name}"

    def compute_scores(
        self,
        weights: Dict[str, torch.Tensor],
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        reference_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算重要性得分。

        参数:
            weights: 模型权重
            gradients: 梯度信息
            reference_weights: 参考权重

        返回:
            重要性得分享典
        """
        if self.scorer.requires_gradients and gradients is None:
            raise ValueError(f"Importance method {self.importance_name!r} requires gradients")
        if self.scorer.requires_reference and reference_weights is None:
            raise ValueError(
                f"Importance method {self.importance_name!r} requires reference_weights"
            )
        raw_scores = self.scorer.score(weights, gradients, reference_weights)
        if not isinstance(raw_scores, Mapping) or any(
            not isinstance(name, str) or not isinstance(score, torch.Tensor)
            for name, score in raw_scores.items()
        ):
            raise TypeError("importance scorers must return a mapping of names to tensors")
        if set(raw_scores) != set(weights):
            missing = sorted(set(weights).difference(raw_scores))
            extra = sorted(set(raw_scores).difference(weights))
            raise ValueError(
                f"importance scorer keys must match weights: missing={missing}, extra={extra}"
            )
        scores: Dict[str, torch.Tensor] = {}
        for name, score in raw_scores.items():
            if score.shape != weights[name].shape:
                raise ValueError(f"Importance score shape must match weight {name!r}")
            if not score.is_floating_point() or score.is_complex():
                raise TypeError(f"Importance score for {name!r} must be real floating point")
            if not torch.isfinite(score).all().item():
                raise ValueError(f"Importance score for {name!r} must contain only finite values")
            scores[name] = score
        return scores

    def compute_layer_ratios(
        self,
        scores: Dict[str, torch.Tensor],
        global_ratio: float,
    ) -> Dict[str, float]:
        """计算每层剪枝率。

        参数:
            scores: 重要性得分享典
            global_ratio: 全局剪枝比例

        返回:
            {layer_name: prune_ratio} 字典
        """
        raw_ratios = self.allocator.allocate(scores, global_ratio)
        if not isinstance(raw_ratios, Mapping) or set(raw_ratios) != set(scores):
            raise ValueError("allocation strategies must return one ratio for every score tensor")
        ratios: Dict[str, float] = {}
        for name, raw_ratio in raw_ratios.items():
            if (
                not isinstance(name, str)
                or isinstance(raw_ratio, bool)
                or not isinstance(raw_ratio, (int, float))
                or not math.isfinite(float(raw_ratio))
                or not 0.0 <= float(raw_ratio) <= 1.0
            ):
                raise ValueError(f"allocation strategy returned an invalid ratio for {name!r}")
            ratios[name] = float(raw_ratio)
        return ratios

    def prune(
        self,
        model: nn.Module,
        scores: Dict[str, torch.Tensor],
        layer_ratios: Dict[str, float],
        device: str = "cpu",
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor], float]:
        """应用剪枝到模型。

        参数:
            model: PyTorch 模型
            scores: 重要性得分享典
            layer_ratios: 每层剪枝率
            device: 设备

        返回:
            (剪枝后的模型，剪枝掩码，实际全局剪枝率)
        """
        return apply_pruning(model, scores, layer_ratios, device)
