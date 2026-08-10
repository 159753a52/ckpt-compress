"""Weibull moment fitting and layer allocation for residual scores."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from scipy.optimize import brentq
from scipy.special import gammaln

from experiments.lib.residual_budget import largest_remainder_counts


WeibullFit = Dict[str, float | int | bool | str]
WeibullFitView = Mapping[str, float | int | bool | str]


def fit_weibull_mom(values: torch.Tensor) -> WeibullFit:
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


def fit_layer_weibull_mom(
    layers: Sequence[Sequence[str]],
    scores: Mapping[str, torch.Tensor],
) -> List[WeibullFit]:
    """Fit one Weibull model per non-empty structural score layer."""
    fits = []
    for layer_index, names in enumerate(layers):
        if not names:
            raise ValueError("Structural layers must be non-empty")
        missing = [name for name in names if name not in scores]
        if missing:
            raise ValueError(f"Missing scores for structural layer: {missing}")
        values = torch.cat([scores[name].flatten() for name in names])
        if values.numel() == 0:
            raise ValueError("Structural layers must contain at least one score")
        fit = fit_weibull_mom(values)
        fit["layer"] = layer_index
        fits.append(fit)
    return fits


def weibull_cdf(
    threshold: float,
    fit: WeibullFitView,
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
    fits: Sequence[WeibullFitView],
    layer_sizes: Sequence[int],
    target: int,
    ratio: float,
    max_layer_ratio: float,
) -> Tuple[List[int], Dict[str, object]]:
    if len(fits) != len(layer_sizes):
        raise ValueError("Fits and layer sizes must have the same length")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in layer_sizes
    ):
        raise ValueError("Layer sizes must be non-negative integers")
    if (
        isinstance(target, bool)
        or not isinstance(target, int)
        or target < 0
        or target > sum(layer_sizes)
    ):
        raise ValueError(
            "target must be a non-negative integer within the layer capacity"
        )
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
                try:
                    numeric = float(fit[key])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Valid Weibull fits must contain numeric {key}"
                    ) from exc
                if not math.isfinite(numeric) or numeric <= 0:
                    raise ValueError(
                        f"Valid Weibull fit {key} must be finite and positive"
                    )
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


__all__ = [
    "fit_layer_weibull_mom",
    "fit_weibull_mom",
    "weibull_cdf",
    "weibull_counts",
]
