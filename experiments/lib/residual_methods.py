"""Method construction and score routing for residual recovery experiments."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch

from experiments.lib.residual_allocation import (
    fit_weibull_mom,
    uniform_counts,
    weibull_counts,
)
from experiments.lib.residual_masks import MaskDict, TensorDict, global_mask, layer_masks
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


def build_masks(
    layers: Sequence[Sequence[str]],
    magnitude_scores: TensorDict,
    components: Dict[str, TensorDict],
    prune_ratio: float,
    max_layer_ratio: float,
    taylor_score_orders: Sequence[torch.Tensor] | None = None,
) -> Tuple[Dict[str, MaskDict], Dict[str, object]]:
    """Build the fixed-order baseline mask family for recovery experiments."""
    taylor_scores = components["taylor"]
    layer_sizes = [sum(taylor_scores[name].numel() for name in layer) for layer in layers]
    eligible_count = sum(layer_sizes)
    target = int(math.floor(prune_ratio * eligible_count))
    uniform_layer_counts = uniform_counts(layer_sizes, target, prune_ratio)

    fits = []
    for index, layer in enumerate(layers):
        values = torch.cat([taylor_scores[name].flatten() for name in layer])
        fit = fit_weibull_mom(values)
        fit["layer"] = index
        fits.append(fit)
    weibull_layer_counts, weibull_metadata = weibull_counts(
        fits, layer_sizes, target, prune_ratio, max_layer_ratio
    )
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
        TAYLOR_WEIBULL_MOM_METHOD: layer_masks(
            layers, taylor_scores, weibull_layer_counts, taylor_score_orders
        ),
        TAYLOR_EXACT_GLOBAL_METHOD: global_mask(eligible_names, taylor_scores, target),
    }
    metadata = {
        "eligible_parameters": eligible_count,
        "target_pruned": target,
        "target_eligible_sparsity": prune_ratio,
        "layer_sizes": layer_sizes,
        "uniform_layer_counts": uniform_layer_counts,
        "weibull_layer_counts": weibull_layer_counts,
        "weibull_fits": fits,
        "weibull": weibull_metadata,
    }
    return masks, metadata


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


__all__ = [
    "build_masks",
    "float_slug",
    "quantile_method_id",
    "score_for_method",
    "spectral_method_id",
]
