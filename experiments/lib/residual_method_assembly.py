"""Adaptive method-family assembly for residual recovery experiments."""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from experiments.lib.residual_calibration import (
    calibrate_spectral_allocation,
    calibrate_trust_region_allocation,
)
from experiments.lib.residual_masks import MaskDict, TensorDict, layer_masks, layer_score_orders
from experiments.lib.residual_method_config import AdaptiveMethodConfig
from experiments.lib.residual_methods import build_masks
from experiments.lib.residual_protocol import (
    TAYLOR_PROBE_TRUST_METHOD,
    quantile_method_id,
    spectral_method_id,
)
from experiments.lib.residual_quantile import calibrate_quantile_smooth_allocation


def assemble_method_family(
    model: torch.nn.Module,
    current_state: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
    layers: Sequence[Sequence[str]],
    magnitude_scores: TensorDict,
    components: Dict[str, TensorDict],
    probe_batches: Sequence[Mapping[str, torch.Tensor]],
    selection_batches: Sequence[Mapping[str, torch.Tensor]],
    config: AdaptiveMethodConfig,
) -> Tuple[Dict[str, MaskDict], Dict[str, Any]]:
    """Build fixed and adaptive masks while preserving calibration side effects."""
    allocation_started = time.perf_counter()
    taylor_score_orders = None
    score_order_seconds = 0.0
    if config.spectral_ranks or config.quantile_smoothness_values:
        score_order_started = time.perf_counter()
        taylor_score_orders = layer_score_orders(
            layers,
            components["taylor"],
            sort_device=config.device,
        )
        score_order_seconds = time.perf_counter() - score_order_started

    masks, metadata = build_masks(
        layers,
        magnitude_scores,
        components,
        config.prune_ratio,
        config.max_layer_ratio,
        taylor_score_orders,
    )
    probe_trust_counts, probe_trust_metadata = calibrate_trust_region_allocation(
        model,
        current_state,
        reference_state,
        layers,
        components["taylor"],
        metadata["uniform_layer_counts"],
        metadata["target_pruned"],
        config.prune_ratio,
        config.max_layer_ratio,
        probe_batches,
        selection_batches,
        config.probe_radius,
        config.trust_radii,
        config.device,
        taylor_score_orders,
    )
    masks[TAYLOR_PROBE_TRUST_METHOD] = layer_masks(
        layers,
        components["taylor"],
        probe_trust_counts,
        taylor_score_orders,
    )
    metadata["probe_trust_layer_counts"] = probe_trust_counts
    metadata["probe_trust"] = probe_trust_metadata

    if config.spectral_ranks:
        spectral_counts, spectral_metadata = calibrate_spectral_allocation(
            model,
            current_state,
            reference_state,
            layers,
            components["taylor"],
            metadata["target_pruned"],
            config.prune_ratio,
            config.max_layer_ratio,
            probe_batches,
            config.spectral_probe_radius,
            config.spectral_ranks,
            config.spectral_trust_radius,
            config.device,
            taylor_score_orders,
        )
        for rank, counts in spectral_counts.items():
            masks[spectral_method_id(rank)] = layer_masks(
                layers,
                components["taylor"],
                counts,
                taylor_score_orders,
            )
        metadata["spectral_layer_counts"] = spectral_counts
        metadata["spectral"] = spectral_metadata

    if config.quantile_smoothness_values:
        if taylor_score_orders is None:
            raise RuntimeError("Quantile allocation requires materialized Taylor score orders")
        quantile_counts, quantile_metadata = calibrate_quantile_smooth_allocation(
            layers,
            components["taylor"],
            taylor_score_orders,
            metadata["target_pruned"],
            config.prune_ratio,
            config.max_layer_ratio,
            config.quantile_trust_radius,
            config.quantile_smoothness_values,
            config.device,
            config.quantile_cost_normalization,
        )
        for smoothness, counts in quantile_counts.items():
            method = quantile_method_id(
                config.quantile_cost_normalization,
                smoothness,
            )
            masks[method] = layer_masks(
                layers,
                components["taylor"],
                counts,
                taylor_score_orders,
            )
        metadata["quantile_smooth_layer_counts"] = {
            str(value): counts for value, counts in quantile_counts.items()
        }
        metadata["quantile_smooth"] = quantile_metadata

    if taylor_score_orders is not None:
        metadata["score_order_seconds"] = score_order_seconds
        metadata["score_order_sort_device"] = config.device
        metadata["score_order_bytes"] = sum(
            order.numel() * order.element_size() for order in taylor_score_orders
        )
    metadata["seconds"] = time.perf_counter() - allocation_started
    metadata["whole_model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return masks, metadata


__all__ = ["AdaptiveMethodConfig", "assemble_method_family"]
