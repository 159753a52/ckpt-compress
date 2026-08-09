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


def largest_remainder_counts(
    real_counts: Sequence[float],
    target: int,
    capacities: Sequence[int],
) -> List[int]:
    counts = [min(int(math.floor(value)), cap) for value, cap in zip(real_counts, capacities)]
    difference = target - sum(counts)
    if difference > 0:
        order = sorted(
            range(len(counts)),
            key=lambda i: (real_counts[i] - math.floor(real_counts[i]), -i),
            reverse=True,
        )
        while difference:
            progressed = False
            for index in order:
                if counts[index] < capacities[index]:
                    counts[index] += 1
                    difference -= 1
                    progressed = True
                    if difference == 0:
                        break
            if not progressed:
                raise RuntimeError("Layer capacities cannot meet the global pruning target")
    elif difference < 0:
        order = sorted(
            range(len(counts)),
            key=lambda i: (real_counts[i] - math.floor(real_counts[i]), -i),
        )
        while difference:
            progressed = False
            for index in order:
                if counts[index] > 0:
                    counts[index] -= 1
                    difference += 1
                    progressed = True
                    if difference == 0:
                        break
            if not progressed:
                raise RuntimeError("Could not round layer counts to the global target")
    return counts


def uniform_counts(layer_sizes: Sequence[int], target: int, ratio: float) -> List[int]:
    real = [ratio * size for size in layer_sizes]
    return largest_remainder_counts(real, target, list(layer_sizes))


def bounded_largest_remainder_counts(
    real_counts: Sequence[float],
    target: int,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
) -> List[int]:
    if not (
        len(real_counts) == len(lower_bounds) == len(upper_bounds)
        and all(lower <= upper for lower, upper in zip(lower_bounds, upper_bounds))
    ):
        raise ValueError("Invalid bounded-rounding inputs")
    if not sum(lower_bounds) <= target <= sum(upper_bounds):
        raise ValueError("Bounds cannot meet the requested global target")

    shifted_real = [
        min(max(value, lower), upper) - lower
        for value, lower, upper in zip(real_counts, lower_bounds, upper_bounds)
    ]
    shifted_target = target - sum(lower_bounds)
    capacities = [upper - lower for lower, upper in zip(lower_bounds, upper_bounds)]
    shifted = largest_remainder_counts(shifted_real, shifted_target, capacities)
    return [value + lower for value, lower in zip(shifted, lower_bounds)]


def trust_region_counts(
    marginal_losses: Sequence[float],
    layer_sizes: Sequence[int],
    target: int,
    ratio: float,
    trust_radius: float,
    max_layer_ratio: float,
) -> List[int]:
    """Project a loss-decreasing rate step onto an exact, bounded budget."""
    if len(marginal_losses) != len(layer_sizes) or not layer_sizes:
        raise ValueError("Marginals and non-empty layer sizes must have the same length")
    if not 0 < trust_radius <= 1:
        raise ValueError("trust_radius must be in (0, 1]")
    if not all(math.isfinite(value) for value in marginal_losses):
        raise ValueError("Marginal losses must be finite")

    lower_rate = max(0.0, ratio - trust_radius)
    upper_rate = min(1.0, max_layer_ratio, ratio + trust_radius)
    lower_bounds = [int(math.ceil(lower_rate * size)) for size in layer_sizes]
    upper_bounds = [int(math.floor(upper_rate * size)) for size in layer_sizes]
    if any(lower > upper for lower, upper in zip(lower_bounds, upper_bounds)):
        raise ValueError("A layer is too small to represent the requested trust region")
    if not sum(lower_bounds) <= target <= sum(upper_bounds):
        raise ValueError("Trust region cannot meet the requested global target")

    # The finite-difference derivative is dL/dp_l. Dividing by n_l gives a
    # comparable per-parameter marginal before the weighted budget projection.
    per_parameter = [
        marginal / size for marginal, size in zip(marginal_losses, layer_sizes)
    ]
    spread = max(per_parameter) - min(per_parameter)
    if spread <= 1e-30:
        return bounded_largest_remainder_counts(
            [ratio * size for size in layer_sizes],
            target,
            lower_bounds,
            upper_bounds,
        )

    # This is the closed-form gradient step for
    #   min_x g^T x + (1 / 2 eta) sum_l n_l x_l^2,
    # followed by a weighted projection onto the budget and box constraints.
    step_size = 2.0 * trust_radius / spread

    def rates_at_shift(shift: float) -> List[float]:
        return [
            min(upper_rate, max(lower_rate, ratio - step_size * value + shift))
            for value in per_parameter
        ]

    low = -2.0
    high = 2.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        total = sum(
            size * rate for size, rate in zip(layer_sizes, rates_at_shift(middle))
        )
        if total < target:
            low = middle
        else:
            high = middle
    rates = rates_at_shift(0.5 * (low + high))
    real_counts = [size * rate for size, rate in zip(layer_sizes, rates)]
    return bounded_largest_remainder_counts(
        real_counts, target, lower_bounds, upper_bounds
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
        counts = uniform_counts(layer_sizes, target, ratio)
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


def budget_tangent_dct_directions(
    layer_sizes: Sequence[int],
    rank: int,
) -> List[List[float]]:
    """Build low-frequency DCT directions orthogonal to the pruning budget."""
    if not layer_sizes or any(size <= 0 for size in layer_sizes):
        raise ValueError("Layer sizes must be positive")
    if not 1 <= rank < len(layer_sizes):
        raise ValueError("rank must be in [1, number of layers)")

    sizes = np.asarray(layer_sizes, dtype=np.float64)
    layer_indices = np.arange(len(layer_sizes), dtype=np.float64)
    directions = []
    for frequency in range(1, rank + 1):
        direction = np.cos(
            math.pi * (layer_indices + 0.5) * frequency / len(layer_sizes)
        )
        direction -= np.dot(sizes, direction) / sizes.sum()
        maximum = np.max(np.abs(direction))
        if maximum <= 1e-12:
            raise ValueError(f"Degenerate spectral direction at frequency {frequency}")
        directions.append((direction / maximum).tolist())
    return directions


def reconstruct_directional_gradient(
    design: Sequence[Sequence[float]],
    responses: Sequence[float],
) -> Tuple[List[float], Dict[str, float | int]]:
    """Return the minimum-norm gradient matching measured directional derivatives."""
    matrix = np.asarray(design, dtype=np.float64)
    values = np.asarray(responses, dtype=np.float64)
    if matrix.ndim != 2 or values.shape != (matrix.shape[0],):
        raise ValueError("Directional design and responses have incompatible shapes")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(values)):
        raise ValueError("Directional measurements must be finite")

    matrix_rank = int(np.linalg.matrix_rank(matrix))
    if matrix_rank != matrix.shape[0]:
        raise ValueError("Directional design must have full row rank")
    condition_number = float(np.linalg.cond(matrix))
    gradient, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
    residual = matrix @ gradient - values
    return gradient.tolist(), {
        "rank": matrix_rank,
        "condition_number": condition_number,
        "response_residual_l2": float(np.linalg.norm(residual)),
    }


def directional_layer_counts(
    direction: Sequence[float],
    layer_sizes: Sequence[int],
    target: int,
    ratio: float,
    probe_radius: float,
    max_layer_ratio: float,
    sign: float,
) -> List[int]:
    if len(direction) != len(layer_sizes):
        raise ValueError("Direction and layer sizes must have the same length")
    lower_bounds = [0] * len(layer_sizes)
    upper_bounds = [int(math.floor(max_layer_ratio * size)) for size in layer_sizes]
    rates = [ratio + sign * probe_radius * value for value in direction]
    if any(rate < 0 or rate > max_layer_ratio for rate in rates):
        raise ValueError("Spectral probe leaves the feasible layer-rate box")
    real_counts = [size * rate for size, rate in zip(layer_sizes, rates)]
    return bounded_largest_remainder_counts(
        real_counts, target, lower_bounds, upper_bounds
    )


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
