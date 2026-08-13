"""Method construction and score routing for residual recovery experiments."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch

from experiments.lib.residual_budget import uniform_counts
from experiments.lib.residual_masks import (
    MaskDict,
    TensorDict,
    global_mask,
    layer_masks,
    layer_rates,
    mask_metrics,
    mask_overlap,
)
from experiments.lib.residual_protocol import (
    FIRST_ORDER_UNIFORM_METHOD,
    RESIDUAL_MAGNITUDE_UNIFORM_METHOD,
    SECOND_ORDER_UNIFORM_METHOD,
    TAYLOR_EXACT_GLOBAL_METHOD,
    TAYLOR_UNIFORM_METHOD,
    TAYLOR_WEIBULL_MOM_METHOD,
    float_slug,
    quantile_method_id,
    spectral_method_id,
)
from experiments.lib.residual_weibull import fit_layer_weibull_mom, weibull_counts


def build_masks(
    layers: Sequence[Sequence[str]],
    magnitude_scores: TensorDict,
    components: Dict[str, TensorDict],
    prune_ratio: float,
    max_layer_ratio: float,
    taylor_score_orders: Sequence[torch.Tensor] | None = None,
    *,
    distributed_moments: bool = False,
) -> Tuple[Dict[str, MaskDict], Dict[str, Any]]:
    """Build the fixed-order baseline mask family for recovery experiments."""
    taylor_scores = components["taylor"]
    weibull_mask, metadata = build_weibull_mask(
        layers,
        taylor_scores,
        prune_ratio,
        max_layer_ratio,
        distributed_moments=distributed_moments,
        score_orders=taylor_score_orders,
    )
    layer_sizes = metadata["layer_sizes"]
    eligible_count = metadata["eligible_parameters"]
    target = metadata["target_pruned"]
    uniform_layer_counts = uniform_counts(layer_sizes, target, prune_ratio)
    eligible_names = [name for layer in layers for name in layer]
    masks = {
        RESIDUAL_MAGNITUDE_UNIFORM_METHOD: layer_masks(
            layers, magnitude_scores, uniform_layer_counts
        ),
        FIRST_ORDER_UNIFORM_METHOD: layer_masks(
            layers, components["first_order"], uniform_layer_counts
        ),
        SECOND_ORDER_UNIFORM_METHOD: layer_masks(
            layers, components["second_order"], uniform_layer_counts
        ),
        TAYLOR_UNIFORM_METHOD: layer_masks(
            layers, taylor_scores, uniform_layer_counts, taylor_score_orders
        ),
        TAYLOR_WEIBULL_MOM_METHOD: weibull_mask,
        TAYLOR_EXACT_GLOBAL_METHOD: global_mask(eligible_names, taylor_scores, target),
    }
    metadata["uniform_layer_counts"] = uniform_layer_counts
    return masks, metadata


def build_weibull_mask(
    layers: Sequence[Sequence[str]],
    taylor_scores: TensorDict,
    prune_ratio: float,
    max_layer_ratio: float,
    *,
    distributed_moments: bool = False,
    score_orders: Sequence[torch.Tensor] | None = None,
) -> Tuple[MaskDict, Dict[str, Any]]:
    """Build only the DACP Weibull mask for memory-constrained paper runs."""
    layer_sizes = [sum(taylor_scores[name].numel() for name in layer) for layer in layers]
    eligible_count = sum(layer_sizes)
    target = int(math.floor(prune_ratio * eligible_count))
    fits = fit_layer_weibull_mom(
        layers,
        taylor_scores,
        distributed=distributed_moments,
    )
    counts, weibull_metadata = weibull_counts(
        fits, layer_sizes, target, prune_ratio, max_layer_ratio
    )
    masks = layer_masks(layers, taylor_scores, counts, score_orders)
    return masks, {
        "eligible_parameters": eligible_count,
        "target_pruned": target,
        "target_eligible_sparsity": prune_ratio,
        "layer_sizes": layer_sizes,
        "weibull_layer_counts": counts,
        "weibull_fits": fits,
        "weibull": weibull_metadata,
    }


def score_for_method(
    method: str,
    magnitude_scores: TensorDict,
    components: Dict[str, TensorDict],
) -> TensorDict:
    """Return the selection score family associated with an experiment method."""
    if method == RESIDUAL_MAGNITUDE_UNIFORM_METHOD:
        return magnitude_scores
    if method == FIRST_ORDER_UNIFORM_METHOD:
        return components["first_order"]
    if method == SECOND_ORDER_UNIFORM_METHOD:
        return components["second_order"]
    return components["taylor"]


def compute_method_diagnostics(
    method: str,
    method_masks: MaskDict,
    exact_masks: MaskDict,
    exact_taylor_proxy_cost: float,
    layers: Sequence[Sequence[str]],
    magnitude_scores: TensorDict,
    components: Dict[str, TensorDict],
    eligible_parameters: int,
) -> Dict[str, Any]:
    """Compute mask-only diagnostics without evaluating or mutating the model."""
    taylor_metrics = mask_metrics(method_masks, components["taylor"])
    metrics: Dict[str, Any] = dict(taylor_metrics)
    selection = mask_metrics(
        method_masks,
        score_for_method(method, magnitude_scores, components),
    )
    metrics["selection_score_cost"] = selection["proxy_cost"]
    metrics["eligible_sparsity"] = taylor_metrics["pruned"] / eligible_parameters
    metrics["layer_rates"] = layer_rates(layers, method_masks)
    metrics["taylor_regret_vs_exact"] = (
        taylor_metrics["proxy_cost"] - exact_taylor_proxy_cost
    ) / max(abs(exact_taylor_proxy_cost), 1e-30)
    metrics["overlap_with_taylor_exact"] = mask_overlap(method_masks, exact_masks)
    return metrics


__all__ = [
    "build_masks",
    "build_weibull_mask",
    "compute_method_diagnostics",
    "float_slug",
    "quantile_method_id",
    "score_for_method",
    "spectral_method_id",
]
