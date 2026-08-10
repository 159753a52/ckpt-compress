"""Weibull moment fitting and layer allocation for residual scores."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from scipy.optimize import brentq
from scipy.special import gammaln

from experiments.lib.residual_budget import largest_remainder_counts


def fit_weibull_mom(values: torch.Tensor) -> Dict[str, float | bool | str]:
    flat = values.detach().float().flatten().cpu()
    count = flat.numel()
    if count == 0:
        return {
            "valid": False,
            "reason": "empty values",
            "zero_fraction": 0.0,
        }
    maximum = flat.max().item()
    zero_fraction = (flat == 0).sum().item() / count
    if not math.isfinite(maximum) or maximum <= 0:
        return {
            "valid": False,
            "reason": "non-positive maximum",
            "zero_fraction": zero_fraction,
        }

    scaled = flat / maximum
    total = scaled.sum(dtype=torch.float64).item()
    total_squared = (scaled * scaled).sum(dtype=torch.float64).item()
    mean = total / count
    variance = max(total_squared / count - mean * mean, 0.0)
    if mean <= 1e-15 or variance <= 0:
        return {
            "valid": False,
            "reason": "degenerate moments",
            "zero_fraction": zero_fraction,
        }
    cv_squared = variance / (mean * mean)
    if cv_squared < 1e-10 or not math.isfinite(cv_squared):
        return {
            "valid": False,
            "reason": "degenerate coefficient of variation",
            "zero_fraction": zero_fraction,
        }

    log_target = math.log1p(cv_squared)

    def objective(shape: float) -> float:
        return float(
            gammaln(1.0 + 2.0 / shape)
            - 2.0 * gammaln(1.0 + 1.0 / shape)
            - log_target
        )

    try:
        shape = brentq(objective, 0.01, 1000.0, maxiter=200)
    except ValueError:
        return {
            "valid": False,
            "reason": "shape root not bracketed",
            "zero_fraction": zero_fraction,
        }
    log_scale = (
        math.log(mean)
        - float(gammaln(1.0 + 1.0 / shape))
        + math.log(maximum)
    )
    scale = math.exp(log_scale)
    if not math.isfinite(scale) or scale <= 0:
        return {
            "valid": False,
            "reason": "invalid scale",
            "zero_fraction": zero_fraction,
        }
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


def weibull_cdf(
    threshold: float,
    fit: Mapping[str, float | bool | str],
) -> float:
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
    capacities = [
        min(size, int(math.floor(max_layer_ratio * size))) for size in layer_sizes
    ]
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


__all__ = ["fit_weibull_mom", "weibull_cdf", "weibull_counts"]
