"""Model-independent allocation math for residual recovery experiments."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import gammaln

from experiments.lib.quantile_allocation import _solve_quantile_smooth_counts
from experiments.lib.residual_budget import (
    bounded_largest_remainder_counts,
    largest_remainder_counts,
    trust_region_counts,
    uniform_counts,
)
from experiments.lib.residual_spectral import (
    budget_tangent_dct_directions,
    directional_layer_counts,
    reconstruct_directional_gradient,
)


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
        flat = torch.cat([scores[name].detach().float().flatten().cpu() for name in names])
        ordered = flat.to(device)[order.to(device)]
        values = ordered.cpu().numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError("Empirical Taylor scores must be finite")
        sorted_scores.append(values)
    score_materialization_seconds = time.perf_counter() - score_started

    uniform_layer_counts = uniform_counts(layer_sizes.tolist(), target, ratio)
    counts_by_smoothness, solver_metadata = _solve_quantile_smooth_counts(
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


def fit_weibull_mom(values: torch.Tensor) -> Dict[str, float | bool | str]:
    flat = values.detach().float().flatten().cpu()
    count = flat.numel()
    maximum = flat.max().item()
    zero_fraction = (flat == 0).sum().item() / count
    if not math.isfinite(maximum) or maximum <= 0:
        return {"valid": False, "reason": "non-positive maximum", "zero_fraction": zero_fraction}

    scaled = flat / maximum
    total = scaled.sum(dtype=torch.float64).item()
    total_squared = (scaled * scaled).sum(dtype=torch.float64).item()
    mean = total / count
    variance = max(total_squared / count - mean * mean, 0.0)
    if mean <= 1e-15 or variance <= 0:
        return {"valid": False, "reason": "degenerate moments", "zero_fraction": zero_fraction}
    cv_squared = variance / (mean * mean)
    if cv_squared < 1e-10 or not math.isfinite(cv_squared):
        return {
            "valid": False,
            "reason": "degenerate coefficient of variation",
            "zero_fraction": zero_fraction,
        }

    log_target = math.log1p(cv_squared)

    def objective(shape: float) -> float:
        return float(gammaln(1.0 + 2.0 / shape) - 2.0 * gammaln(1.0 + 1.0 / shape) - log_target)

    try:
        shape = brentq(objective, 0.01, 1000.0, maxiter=200)
    except ValueError:
        return {
            "valid": False,
            "reason": "shape root not bracketed",
            "zero_fraction": zero_fraction,
        }
    log_scale = math.log(mean) - float(gammaln(1.0 + 1.0 / shape)) + math.log(maximum)
    scale = math.exp(log_scale)
    if not math.isfinite(scale) or scale <= 0:
        return {"valid": False, "reason": "invalid scale", "zero_fraction": zero_fraction}
    return {
        "valid": True,
        "count": count,
        "mean": mean * maximum,
        "variance": variance * maximum * maximum,
        "cv_squared": cv_squared,
        "shape": shape,
        "scale": scale,
        "zero_fraction": zero_fraction,
    }


def weibull_cdf(threshold: float, fit: Mapping[str, float | bool | str]) -> float:
    if threshold <= 0:
        return 0.0
    shape = float(fit["shape"])
    scale = float(fit["scale"])
    log_power = shape * (math.log(threshold) - math.log(scale))
    if log_power > 40:
        return 1.0
    if log_power < -40:
        return math.exp(log_power)
    return -math.expm1(-math.exp(log_power))


def weibull_counts(
    fits: Sequence[Mapping[str, float | bool | str]],
    layer_sizes: Sequence[int],
    target: int,
    ratio: float,
    max_layer_ratio: float,
) -> Tuple[List[int], Dict[str, object]]:
    if len(fits) != len(layer_sizes):
        raise ValueError("Fits and layer sizes must have the same length")
    capacities = [min(size, int(math.floor(max_layer_ratio * size))) for size in layer_sizes]
    if sum(capacities) < target:
        raise ValueError("max_layer_ratio makes the requested global ratio infeasible")

    def real_counts_at(threshold: float) -> List[float]:
        result = []
        for fit, size, capacity in zip(fits, layer_sizes, capacities):
            if bool(fit["valid"]):
                count = size * weibull_cdf(threshold, fit)
            else:
                count = size * ratio
            result.append(min(count, float(capacity)))
        return result

    valid_scales = [float(fit["scale"]) for fit in fits if bool(fit["valid"])]
    if not valid_scales:
        counts = largest_remainder_counts(
            [ratio * size for size in layer_sizes],
            target,
            capacities,
        )
        return counts, {"threshold": None, "fallback": "all Weibull fits invalid"}

    high = max(valid_scales)
    while sum(real_counts_at(high)) < target:
        high *= 2.0
    threshold = brentq(
        lambda value: sum(real_counts_at(value)) - target,
        0.0,
        high,
        maxiter=200,
    )
    real_counts = real_counts_at(threshold)
    counts = largest_remainder_counts(real_counts, target, capacities)
    return counts, {
        "threshold": threshold,
        "real_counts": real_counts,
        "capacities": capacities,
        "fallback": None,
    }


__all__ = [
    "bounded_largest_remainder_counts",
    "budget_tangent_dct_directions",
    "calibrate_quantile_smooth_allocation",
    "directional_layer_counts",
    "fit_weibull_mom",
    "largest_remainder_counts",
    "reconstruct_directional_gradient",
    "trust_region_counts",
    "uniform_counts",
    "weibull_cdf",
    "weibull_counts",
]
