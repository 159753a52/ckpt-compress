"""Tensor adapter for empirical quantile residual allocation."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from experiments.lib.quantile_allocation import solve_quantile_smooth_counts
from experiments.lib.residual_budget import uniform_counts


def calibrate_quantile_smooth_allocation(
    layers: Sequence[Sequence[str]],
    scores: Mapping[str, torch.Tensor],
    score_orders: Sequence[torch.Tensor],
    target: int,
    ratio: float,
    max_layer_ratio: float,
    trust_radius: float,
    smoothness_values: Sequence[float],
    device: str,
    normalization: str = "global",
    max_iterations: int = 300,
    tolerance: float = 1e-9,
) -> Tuple[Dict[float, List[int]], Dict[str, object]]:
    """Materialize Taylor curves and run the pure NumPy quantile solver."""
    if len(score_orders) != len(layers) or not layers:
        raise ValueError("Score orders must match non-empty structural layers")
    unique_smoothness = sorted(set(float(value) for value in smoothness_values))
    if not unique_smoothness or any(
        not math.isfinite(value) or value <= 0 for value in unique_smoothness
    ):
        raise ValueError("Smoothness values must be finite and positive")
    if not 0 < trust_radius <= min(ratio, max_layer_ratio - ratio):
        raise ValueError("trust_radius must stay inside the feasible layer-rate box")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("Solver iteration count and tolerance must be positive")
    if normalization not in {"global", "layer_uniform_cost"}:
        raise ValueError("Unsupported empirical-cost normalization")

    started = time.perf_counter()
    layer_sizes = np.asarray(
        [sum(scores[name].numel() for name in names) for names in layers],
        dtype=np.int64,
    )
    lower_rate = max(0.0, ratio - trust_radius)
    upper_rate = min(max_layer_ratio, ratio + trust_radius)
    lower_bounds = np.ceil(lower_rate * layer_sizes).astype(np.int64)
    upper_bounds = np.floor(upper_rate * layer_sizes).astype(np.int64)
    if not int(lower_bounds.sum()) <= target <= int(upper_bounds.sum()):
        raise ValueError("Quantile smoothness bounds cannot meet the exact budget")

    score_started = time.perf_counter()
    sorted_scores = []
    for names, order, size in zip(layers, score_orders, layer_sizes):
        if order.numel() != int(size):
            raise ValueError("A score order has the wrong number of elements")
        flat = torch.cat(
            [scores[name].detach().float().flatten().cpu() for name in names]
        )
        ordered = flat.to(device)[order.to(device)]
        values = ordered.cpu().numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError("Empirical Taylor scores must be finite")
        sorted_scores.append(values)
    score_materialization_seconds = time.perf_counter() - score_started

    uniform_layer_counts = uniform_counts(layer_sizes.tolist(), target, ratio)
    counts_by_smoothness, solver_metadata = solve_quantile_smooth_counts(
        sorted_scores=sorted_scores,
        layer_sizes=layer_sizes,
        uniform_layer_counts=uniform_layer_counts,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        target=target,
        trust_radius=trust_radius,
        smoothness_values=unique_smoothness,
        normalization=normalization,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    solutions = solver_metadata.pop("solutions")
    return counts_by_smoothness, {
        "seconds": time.perf_counter() - started,
        "score_materialization_seconds": score_materialization_seconds,
        **solver_metadata,
        "model_forward_evaluations": 0,
        "batch_forward_evaluations": 0,
        "uses_validation_data": False,
        "solutions": solutions,
    }


__all__ = ["calibrate_quantile_smooth_allocation"]
