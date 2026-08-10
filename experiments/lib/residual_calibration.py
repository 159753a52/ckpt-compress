"""Loss-probe calibration strategies for residual allocation experiments."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from experiments.lib.residual_allocation import (
    budget_tangent_dct_directions,
    directional_layer_counts,
    reconstruct_directional_gradient,
    trust_region_counts,
)
from experiments.lib.residual_masks import (
    apply_layer_mask,
    apply_mask_from_device_states,
    cache_mask_states_on_device,
    layer_mask_at_count,
    layer_masks,
    restore_with_mask,
)
from experiments.lib.residual_runtime import lm_loss, synchronize_device


def batch_loss_values(
    model: nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    device: str,
) -> List[float]:
    model.eval()
    with torch.no_grad():
        return [lm_loss(model, batch, device).item() for batch in batches]


def calibrate_trust_region_allocation(
    model: nn.Module,
    current_state: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
    layers: Sequence[Sequence[str]],
    scores: Mapping[str, torch.Tensor],
    uniform_layer_counts: Sequence[int],
    target: int,
    ratio: float,
    max_layer_ratio: float,
    probe_batches: Sequence[Mapping[str, torch.Tensor]],
    selection_batches: Sequence[Mapping[str, torch.Tensor]],
    probe_radius: float,
    candidate_trust_radii: Sequence[float],
    device: str,
    score_orders: Sequence[torch.Tensor] | None = None,
) -> Tuple[List[int], Dict[str, object]]:
    """Estimate true-loss marginals near uniform and select a bounded rate step."""
    if not probe_batches or not selection_batches:
        raise ValueError("Probe and selection batches must both be non-empty")
    if not 0 < probe_radius < min(ratio, 1.0 - ratio):
        raise ValueError("probe_radius must fit strictly around the uniform rate")

    started = time.perf_counter()
    layer_sizes = [sum(scores[name].numel() for name in names) for names in layers]
    uniform_masks = layer_masks(layers, scores, uniform_layer_counts, score_orders)
    restore_with_mask(model, current_state, reference_state, uniform_masks, device)
    uniform_probe_losses = batch_loss_values(model, probe_batches, device)
    uniform_selection_losses = batch_loss_values(model, selection_batches, device)

    marginals = []
    probes = []
    for index, (names, size, base_count) in enumerate(
        zip(layers, layer_sizes, uniform_layer_counts)
    ):
        count_delta = max(1, int(round(probe_radius * size)))
        lower_count = max(0, base_count - count_delta)
        upper_count = min(
            int(math.floor(max_layer_ratio * size)),
            base_count + count_delta,
        )
        if lower_count >= upper_count:
            raise ValueError(
                f"Layer {index} has no room for a finite-difference probe"
            )

        score_order = score_orders[index] if score_orders is not None else None
        upper_masks = layer_mask_at_count(
            names,
            scores,
            upper_count,
            score_order,
        )
        apply_layer_mask(
            model,
            current_state,
            reference_state,
            upper_masks,
            device,
        )
        upper_losses = batch_loss_values(model, probe_batches, device)

        lower_masks = layer_mask_at_count(
            names,
            scores,
            lower_count,
            score_order,
        )
        apply_layer_mask(
            model,
            current_state,
            reference_state,
            lower_masks,
            device,
        )
        lower_losses = batch_loss_values(model, probe_batches, device)

        base_layer_masks = {name: uniform_masks[name] for name in names}
        apply_layer_mask(
            model,
            current_state,
            reference_state,
            base_layer_masks,
            device,
        )
        lower_rate = lower_count / size
        upper_rate = upper_count / size
        paired_derivatives = [
            (upper - lower) / (upper_rate - lower_rate)
            for upper, lower in zip(upper_losses, lower_losses)
        ]
        marginal = float(np.mean(paired_derivatives))
        marginals.append(marginal)
        probes.append(
            {
                "layer": index,
                "lower_count": lower_count,
                "upper_count": upper_count,
                "lower_rate": lower_rate,
                "upper_rate": upper_rate,
                "lower_mean_loss": float(np.mean(lower_losses)),
                "upper_mean_loss": float(np.mean(upper_losses)),
                "marginal_loss": marginal,
                "marginal_standard_error": float(
                    np.std(paired_derivatives, ddof=1)
                    / math.sqrt(len(paired_derivatives))
                )
                if len(paired_derivatives) > 1
                else None,
            }
        )
        del upper_masks, lower_masks

    candidates = [
        {
            "trust_radius": 0.0,
            "counts": list(uniform_layer_counts),
            "mean_selection_loss": float(np.mean(uniform_selection_losses)),
            "paired_delta_vs_uniform": 0.0,
        }
    ]
    for trust_radius in sorted(set(candidate_trust_radii)):
        counts = trust_region_counts(
            marginals,
            layer_sizes,
            target,
            ratio,
            trust_radius,
            max_layer_ratio,
        )
        candidate_masks = layer_masks(layers, scores, counts, score_orders)
        restore_with_mask(
            model,
            current_state,
            reference_state,
            candidate_masks,
            device,
        )
        losses = batch_loss_values(model, selection_batches, device)
        deltas = [
            candidate - baseline
            for candidate, baseline in zip(losses, uniform_selection_losses)
        ]
        candidates.append(
            {
                "trust_radius": trust_radius,
                "counts": counts,
                "mean_selection_loss": float(np.mean(losses)),
                "paired_delta_vs_uniform": float(np.mean(deltas)),
                "paired_delta_standard_error": float(
                    np.std(deltas, ddof=1) / math.sqrt(len(deltas))
                )
                if len(deltas) > 1
                else None,
            }
        )
        del candidate_masks

    selected = min(
        candidates,
        key=lambda item: (
            item["mean_selection_loss"],
            item["trust_radius"],
        ),
    )
    restore_with_mask(
        model,
        current_state,
        reference_state,
        uniform_masks,
        device,
    )
    metadata = {
        "seconds": time.perf_counter() - started,
        "probe_radius": probe_radius,
        "probe_batches": len(probe_batches),
        "selection_batches": len(selection_batches),
        "uniform_probe_mean_loss": float(np.mean(uniform_probe_losses)),
        "uniform_selection_mean_loss": float(np.mean(uniform_selection_losses)),
        "probes": probes,
        "marginal_losses": marginals,
        "candidates": candidates,
        "selected_trust_radius": selected["trust_radius"],
        "selected_counts": selected["counts"],
        "selection_uses_validation_data": False,
    }
    return list(selected["counts"]), metadata


def calibrate_spectral_allocation(
    model: nn.Module,
    current_state: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
    layers: Sequence[Sequence[str]],
    scores: Mapping[str, torch.Tensor],
    target: int,
    ratio: float,
    max_layer_ratio: float,
    probe_batches: Sequence[Mapping[str, torch.Tensor]],
    probe_radius: float,
    ranks: Sequence[int],
    trust_radius: float,
    device: str,
    score_orders: Sequence[torch.Tensor] | None = None,
) -> Tuple[Dict[int, List[int]], Dict[str, object]]:
    """Estimate a low-rank rate gradient from budget-preserving spectral probes."""
    if not probe_batches:
        raise ValueError("Spectral probing requires at least one batch")
    if not 0 < ratio < max_layer_ratio <= 1:
        raise ValueError("Spectral probing requires 0 < ratio < max_layer_ratio <= 1")
    if not 0 < probe_radius <= min(ratio, max_layer_ratio - ratio):
        raise ValueError("probe_radius must stay inside the feasible layer-rate box")
    unique_ranks = sorted(set(ranks))
    if not unique_ranks:
        raise ValueError("At least one spectral rank is required")
    if unique_ranks[0] < 1 or unique_ranks[-1] >= len(layers):
        raise ValueError("Spectral ranks must be in [1, number of layers)")

    started = time.perf_counter()
    layer_sizes = [sum(scores[name].numel() for name in names) for names in layers]
    if abs(target - ratio * sum(layer_sizes)) > 1.0:
        raise ValueError("target must match ratio times the eligible parameter count")

    directions = budget_tangent_dct_directions(layer_sizes, max(unique_ranks))
    design = []
    responses = []
    measurements = []
    mask_seconds = 0.0
    evaluation_seconds = 0.0
    cache_started = time.perf_counter()
    eligible_names = [name for names in layers for name in names]
    model.load_state_dict(current_state, strict=True)
    current_device, reference_device = cache_mask_states_on_device(
        current_state,
        reference_state,
        eligible_names,
        device,
    )
    synchronize_device(device)
    device_cache_seconds = time.perf_counter() - cache_started
    device_cache_bytes = sum(
        tensor.numel() * tensor.element_size()
        for state in (current_device, reference_device)
        for tensor in state.values()
    )

    for index, direction in enumerate(directions, start=1):
        plus_counts = directional_layer_counts(
            direction,
            layer_sizes,
            target,
            ratio,
            probe_radius,
            max_layer_ratio,
            1.0,
        )
        minus_counts = directional_layer_counts(
            direction,
            layer_sizes,
            target,
            ratio,
            probe_radius,
            max_layer_ratio,
            -1.0,
        )

        mask_started = time.perf_counter()
        plus_masks = layer_masks(layers, scores, plus_counts, score_orders)
        mask_seconds += time.perf_counter() - mask_started
        evaluation_started = time.perf_counter()
        apply_mask_from_device_states(
            model,
            current_device,
            reference_device,
            plus_masks,
            device,
        )
        plus_losses = batch_loss_values(model, probe_batches, device)
        evaluation_seconds += time.perf_counter() - evaluation_started
        del plus_masks

        mask_started = time.perf_counter()
        minus_masks = layer_masks(layers, scores, minus_counts, score_orders)
        mask_seconds += time.perf_counter() - mask_started
        evaluation_started = time.perf_counter()
        apply_mask_from_device_states(
            model,
            current_device,
            reference_device,
            minus_masks,
            device,
        )
        minus_losses = batch_loss_values(model, probe_batches, device)
        evaluation_seconds += time.perf_counter() - evaluation_started
        del minus_masks

        plus_rates = [
            count / size for count, size in zip(plus_counts, layer_sizes)
        ]
        minus_rates = [
            count / size for count, size in zip(minus_counts, layer_sizes)
        ]
        actual_direction = [
            (plus - minus) / (2.0 * probe_radius)
            for plus, minus in zip(plus_rates, minus_rates)
        ]
        paired_derivatives = [
            (plus - minus) / (2.0 * probe_radius)
            for plus, minus in zip(plus_losses, minus_losses)
        ]
        response = float(np.mean(paired_derivatives))
        design.append(actual_direction)
        responses.append(response)
        measurements.append(
            {
                "frequency": index,
                "plus_counts": plus_counts,
                "minus_counts": minus_counts,
                "actual_direction": actual_direction,
                "response": response,
                "response_standard_error": float(
                    np.std(paired_derivatives, ddof=1)
                    / math.sqrt(len(paired_derivatives))
                )
                if len(paired_derivatives) > 1
                else None,
            }
        )

    counts_by_rank = {}
    reconstructions = {}
    for rank in unique_ranks:
        gradient, reconstruction = reconstruct_directional_gradient(
            design[:rank],
            responses[:rank],
        )
        counts = trust_region_counts(
            gradient,
            layer_sizes,
            target,
            ratio,
            trust_radius,
            max_layer_ratio,
        )
        counts_by_rank[rank] = counts
        reconstruction["gradient"] = gradient
        reconstruction["layer_counts"] = counts
        reconstructions[str(rank)] = reconstruction

    with torch.no_grad():
        named_params = dict(model.named_parameters())
        for name, value in current_device.items():
            named_params[name].copy_(value)
    synchronize_device(device)
    del current_device, reference_device
    return counts_by_rank, {
        "seconds": time.perf_counter() - started,
        "mask_seconds": mask_seconds,
        "evaluation_seconds": evaluation_seconds,
        "device_cache_seconds": device_cache_seconds,
        "device_cache_bytes": device_cache_bytes,
        "probe_batches": len(probe_batches),
        "probe_radius": probe_radius,
        "trust_radius": trust_radius,
        "direction_evaluations": 2 * len(directions),
        "batch_forward_evaluations": 2 * len(directions) * len(probe_batches),
        "measurements": measurements,
        "reconstructions": reconstructions,
        "returned_model_state": "current_uncompressed",
    }


__all__ = [
    "batch_loss_values",
    "calibrate_spectral_allocation",
    "calibrate_trust_region_allocation",
]
