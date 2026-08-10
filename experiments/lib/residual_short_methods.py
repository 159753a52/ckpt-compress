"""Pure problem construction and method diagnostics for the short gate."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import torch
import torch.nn as nn

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
    RESIDUAL_MAGNITUDE_UNIFORM_METHOD,
    SHORT_GATE_METHODS,
    TAYLOR_EXACT_GLOBAL_METHOD,
    TAYLOR_UNIFORM_METHOD,
    TAYLOR_WEIBULL_MOM_METHOD,
)
from experiments.lib.residual_scoring import eligible_layers
from experiments.lib.residual_weibull import fit_weibull_mom, weibull_counts


@dataclass(frozen=True)
class ShortResidualScope:
    """Residual tensors and exact parameter scope shared by all short-gate methods."""

    layers: List[List[str]]
    eligible_names: List[str]
    layer_sizes: List[int]
    eligible_count: int
    model_count: int
    target_pruned: int
    delta: TensorDict
    magnitude_scores: TensorDict

    def to_result_dict(self, prune_ratio: float) -> Dict[str, object]:
        return {
            "transformer_layers": len(self.layers),
            "eligible_tensors": len(self.eligible_names),
            "eligible_parameters": self.eligible_count,
            "whole_model_parameters": self.model_count,
            "target_pruned": self.target_pruned,
            "target_eligible_sparsity": prune_ratio,
            "target_whole_model_sparsity": self.target_pruned / self.model_count,
            "layer_sizes": self.layer_sizes,
            "eligible_names": self.eligible_names,
        }


def build_short_residual_scope(
    model: nn.Module,
    current_state: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
    prune_ratio: float,
) -> ShortResidualScope:
    """Materialize the checkpoint residual over the gate's eligible tensors."""
    layers = eligible_layers(model)
    eligible_names = [name for layer in layers for name in layer]
    layer_sizes = [sum(current_state[name].numel() for name in layer) for layer in layers]
    eligible_count = sum(layer_sizes)
    model_count = sum(parameter.numel() for parameter in model.parameters())
    target_pruned = int(math.floor(prune_ratio * eligible_count))
    delta = {
        name: current_state[name].detach().float() - reference_state[name].detach().float()
        for name in eligible_names
    }
    return ShortResidualScope(
        layers=layers,
        eligible_names=eligible_names,
        layer_sizes=layer_sizes,
        eligible_count=eligible_count,
        model_count=model_count,
        target_pruned=target_pruned,
        delta=delta,
        magnitude_scores={name: values.abs() for name, values in delta.items()},
    )


def build_short_gate_masks(
    scope: ShortResidualScope,
    taylor_scores: Mapping[str, torch.Tensor],
    prune_ratio: float,
    max_layer_ratio: float,
) -> Tuple[Dict[str, MaskDict], Dict[str, object]]:
    """Fit allocations and construct the short gate's four equal-budget masks."""
    started = time.perf_counter()
    fits = []
    for index, layer in enumerate(scope.layers):
        values = torch.cat([taylor_scores[name].flatten() for name in layer])
        fit = fit_weibull_mom(values)
        fit["layer"] = index
        fits.append(fit)

    uniform_layer_counts = uniform_counts(
        scope.layer_sizes,
        scope.target_pruned,
        prune_ratio,
    )
    weibull_layer_counts, weibull_allocation = weibull_counts(
        fits,
        scope.layer_sizes,
        scope.target_pruned,
        prune_ratio,
        max_layer_ratio,
    )
    masks = {
        RESIDUAL_MAGNITUDE_UNIFORM_METHOD: layer_masks(
            scope.layers,
            scope.magnitude_scores,
            uniform_layer_counts,
        ),
        TAYLOR_UNIFORM_METHOD: layer_masks(
            scope.layers,
            taylor_scores,
            uniform_layer_counts,
        ),
        TAYLOR_WEIBULL_MOM_METHOD: layer_masks(
            scope.layers,
            taylor_scores,
            weibull_layer_counts,
        ),
        TAYLOR_EXACT_GLOBAL_METHOD: global_mask(
            scope.eligible_names,
            taylor_scores,
            scope.target_pruned,
        ),
    }
    if tuple(masks) != SHORT_GATE_METHODS:
        raise RuntimeError("Short-gate method order drifted from the result protocol")
    return masks, {
        "seconds": time.perf_counter() - started,
        "weibull_fits": fits,
        "uniform_layer_counts": uniform_layer_counts,
        "weibull_layer_counts": weibull_layer_counts,
        "weibull": weibull_allocation,
    }


def diagnose_short_gate_masks(
    scope: ShortResidualScope,
    masks: Mapping[str, MaskDict],
    taylor_scores: Mapping[str, torch.Tensor],
) -> Dict[str, Dict[str, object]]:
    """Compute the established proxy, sparsity, layer-rate, and overlap schema."""
    if tuple(masks) != SHORT_GATE_METHODS:
        raise ValueError("Short-gate masks must follow the result method protocol")
    exact_metrics = mask_metrics(masks[TAYLOR_EXACT_GLOBAL_METHOD], taylor_scores)
    exact_proxy_cost = float(exact_metrics["proxy_cost"])
    diagnostics: Dict[str, Dict[str, object]] = {}
    for method, method_masks in masks.items():
        metrics: Dict[str, object] = mask_metrics(method_masks, taylor_scores)
        pruned = int(metrics["pruned"])
        proxy_cost = float(metrics["proxy_cost"])
        metrics["eligible_sparsity"] = pruned / scope.eligible_count
        metrics["whole_model_sparsity"] = pruned / scope.model_count
        metrics["layer_rates"] = layer_rates(scope.layers, method_masks)
        metrics["additive_regret_vs_exact"] = (
            (proxy_cost - exact_proxy_cost) / max(abs(exact_proxy_cost), 1e-30)
        )
        metrics["overlap_with_exact"] = mask_overlap(
            method_masks,
            masks[TAYLOR_EXACT_GLOBAL_METHOD],
        )
        diagnostics[method] = metrics
    return diagnostics


__all__ = [
    "SHORT_GATE_METHODS",
    "ShortResidualScope",
    "build_short_gate_masks",
    "build_short_residual_scope",
    "diagnose_short_gate_masks",
]
