"""Weibull moment fitting and layer allocation for residual scores."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from scipy.optimize import brentq
from scipy.special import gammaln

from experiments.lib.distributed_stats import (
    ScoreMoments,
    layer_score_moments,
    reduce_score_moments,
    score_moments,
)
from experiments.lib.residual_budget import largest_remainder_counts

WeibullFit = Dict[str, object]
WeibullFitView = Mapping[str, object]


def _positive_fit_parameter(fit: WeibullFitView, key: str) -> float:
    value = fit.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Valid Weibull fits must contain numeric {key}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"Valid Weibull fit {key} must be finite and positive")
    return numeric


def fit_weibull_from_moments(moments: ScoreMoments) -> WeibullFit:
    """Fit a Weibull distribution from streamable sufficient statistics."""
    count = moments.count
    if count == 0:
        return {
            "valid": False,
            "reason": "empty values",
            "count": 0,
            "zero_fraction": 0.0,
        }
    maximum = moments.maximum
    zero_fraction = moments.zero_count / count
    mean = moments.total / count
    variance = max(moments.total_squared / count - mean * mean, 0.0)
    cv_squared = variance / (mean * mean) if mean > 0 else 0.0
    evidence = {
        "count": count,
        "mean": mean,
        "variance": variance,
        "cv_squared": cv_squared,
        "maximum": maximum,
        "zero_fraction": zero_fraction,
    }
    if not math.isfinite(maximum) or maximum <= 0:
        return {
            "valid": False,
            "reason": "non-positive maximum",
            **evidence,
        }

    if mean <= 1e-15 or variance <= 0:
        return {
            "valid": False,
            "reason": "degenerate moments",
            **evidence,
        }
    if cv_squared < 1e-10 or not math.isfinite(cv_squared):
        return {
            "valid": False,
            "reason": "degenerate coefficient of variation",
            **evidence,
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
            **evidence,
        }
    log_scale = math.log(mean) - float(gammaln(1.0 + 1.0 / shape))
    scale = math.exp(log_scale)
    if not math.isfinite(scale) or scale <= 0:
        return {
            "valid": False,
            "reason": "invalid scale",
            **evidence,
        }
    return {
        "valid": True,
        "count": count,
        "mean": mean,
        "variance": variance,
        "cv_squared": cv_squared,
        "maximum": maximum,
        "shape": shape,
        "scale": scale,
        "zero_fraction": zero_fraction,
    }


def fit_weibull_mom(values: torch.Tensor) -> WeibullFit:
    return fit_weibull_from_moments(score_moments(values))


def fit_layer_weibull_mom(
    layers: Sequence[Sequence[str]],
    scores: Mapping[str, torch.Tensor],
    *,
    distributed: bool = False,
    process_group=None,
) -> List[WeibullFit]:
    """Fit one Weibull model per non-empty structural score layer."""
    moments = layer_score_moments(layers, scores)
    reduction: Dict[str, object] = {
        "distributed": False,
        "world_size": 1,
        "backend": None,
        "communicated_scalars_per_rank": 0,
    }
    if distributed:
        moments, reduction = reduce_score_moments(moments, process_group)
    fits: List[WeibullFit] = []
    for layer_index, layer_moments in enumerate(moments):
        fit = fit_weibull_from_moments(layer_moments)
        fit["layer"] = layer_index
        fit["moment_reduction"] = reduction
        fits.append(fit)
    return fits


def weibull_cdf(
    threshold: float,
    fit: WeibullFitView,
) -> float:
    if threshold <= 0:
        return 0.0
    shape = _positive_fit_parameter(fit, "shape")
    scale = _positive_fit_parameter(fit, "scale")
    log_power = shape * (math.log(threshold) - math.log(scale))
    if log_power > 40:
        return 1.0
    if log_power < -40:
        return math.exp(log_power)
    return -math.expm1(-math.exp(log_power))


def weibull_counts(
    fits: Sequence[WeibullFitView],
    layer_sizes: Sequence[int],
    target: int,
    ratio: float,
    max_layer_ratio: float,
) -> Tuple[List[int], Dict[str, object]]:
    if len(fits) != len(layer_sizes):
        raise ValueError("Fits and layer sizes must have the same length")
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in layer_sizes):
        raise ValueError("Layer sizes must be non-negative integers")
    if (
        isinstance(target, bool)
        or not isinstance(target, int)
        or target < 0
        or target > sum(layer_sizes)
    ):
        raise ValueError("target must be a non-negative integer within the layer capacity")
    normalized_ratios = {}
    for name, value in (("ratio", ratio), ("max_layer_ratio", max_layer_ratio)):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number in [0, 1]")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number in [0, 1]") from exc
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{name} must be a finite number in [0, 1]")
        normalized_ratios[name] = numeric
    ratio = normalized_ratios["ratio"]
    max_layer_ratio = normalized_ratios["max_layer_ratio"]
    for fit in fits:
        if not isinstance(fit, Mapping) or "valid" not in fit:
            raise ValueError("Each Weibull fit must contain a valid flag")
        if bool(fit["valid"]):
            for key in ("shape", "scale"):
                _positive_fit_parameter(fit, key)
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

    valid_scales = [_positive_fit_parameter(fit, "scale") for fit in fits if bool(fit["valid"])]
    if not valid_scales:
        counts = largest_remainder_counts(
            [ratio * size for size in layer_sizes],
            target,
            capacities,
        )
        return counts, {"threshold": None, "fallback": "all Weibull fits invalid"}

    minimum_real_counts = real_counts_at(0.0)
    if sum(minimum_real_counts) >= target:
        counts = largest_remainder_counts(minimum_real_counts, target, capacities)
        return counts, {
            "threshold": 0.0,
            "real_counts": minimum_real_counts,
            "capacities": capacities,
            "fallback": None,
        }

    reference_scale = max(valid_scales)
    high_multiplier = 1.0
    while sum(real_counts_at(reference_scale * high_multiplier)) < target:
        high_multiplier *= 2.0
    threshold_multiplier = brentq(
        lambda multiplier: sum(real_counts_at(reference_scale * multiplier)) - target,
        0.0,
        high_multiplier,
        maxiter=200,
    )
    threshold = reference_scale * threshold_multiplier
    real_counts = real_counts_at(threshold)
    counts = largest_remainder_counts(real_counts, target, capacities)
    return counts, {
        "threshold": threshold,
        "real_counts": real_counts,
        "capacities": capacities,
        "fallback": None,
    }


__all__ = [
    "fit_layer_weibull_mom",
    "fit_weibull_from_moments",
    "fit_weibull_mom",
    "weibull_cdf",
    "weibull_counts",
]
