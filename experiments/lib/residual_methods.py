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
        "residual_magnitude_uniform": layer_masks(
            layers, magnitude_scores, uniform_layer_counts
        ),
        "first_order_uniform": layer_masks(
            layers, components["first_order"], uniform_layer_counts
        ),
        "second_order_uniform": layer_masks(
            layers, components["second_order"], uniform_layer_counts
        ),
        "taylor_uniform": layer_masks(
            layers, taylor_scores, uniform_layer_counts, taylor_score_orders
        ),
        "taylor_weibull_mom": layer_masks(
            layers, taylor_scores, weibull_layer_counts, taylor_score_orders
        ),
        "taylor_exact_global": global_mask(eligible_names, taylor_scores, target),
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
    if method == "residual_magnitude_uniform":
        return magnitude_scores
    if method == "first_order_uniform":
        return components["first_order"]
    if method == "second_order_uniform":
        return components["second_order"]
    return components["taylor"]


def float_slug(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p").replace("+", "")


__all__ = ["build_masks", "float_slug", "score_for_method"]
