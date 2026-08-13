"""Gamma method-of-moments allocation for sensitivity experiments."""

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
from scipy import stats
from scipy.optimize import bisect


@dataclass(frozen=True)
class GammaLayerFit:
    name: str
    size: int
    positive_count: int
    shape: Optional[float]
    scale: Optional[float]
    fallback_reason: Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.shape is not None and self.scale is not None

    def parameters(self) -> Tuple[float, float]:
        """Return a validated fit pair for numeric allocation code."""
        if self.shape is None or self.scale is None:
            raise ValueError(f"Gamma layer {self.name!r} has no valid fit parameters")
        return self.shape, self.scale


@dataclass(frozen=True)
class GammaAllocationProblem:
    layers: Tuple[GammaLayerFit, ...]
    total_parameters: int


def fit_gamma_mom_problem(
    scores: Mapping[str, torch.Tensor],
    *,
    min_positive: int = 10,
    moment_floor: float = 1e-12,
    shape_bounds: Tuple[float, float] = (0.01, 1000.0),
    scale_bounds: Tuple[float, float] = (0.01, 1000.0),
) -> GammaAllocationProblem:
    """Fit per-layer Gamma parameters with the legacy MoM and clamp rules."""
    if min_positive < 1:
        raise ValueError(f"min_positive must be positive, got {min_positive}")
    if moment_floor <= 0:
        raise ValueError(f"moment_floor must be positive, got {moment_floor}")
    for label, bounds in (("shape", shape_bounds), ("scale", scale_bounds)):
        if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1]:
            raise ValueError(f"{label}_bounds must satisfy 0 < lower <= upper")

    layers = []
    for name, score in scores.items():
        data = score.detach().flatten().float().cpu()
        positive = data[data > 0]
        positive_count = positive.numel()
        reason = None
        shape = scale = None
        if positive_count < min_positive:
            reason = "insufficient_positive"
        else:
            mean = positive.mean().item()
            variance = positive.var().item()
            if not math.isfinite(mean) or not math.isfinite(variance):
                reason = "non_finite_moments"
            elif variance < moment_floor or mean < moment_floor:
                reason = "degenerate_moments"
            else:
                shape = max(shape_bounds[0], min(mean**2 / variance, shape_bounds[1]))
                scale = max(scale_bounds[0], min(variance / mean, scale_bounds[1]))

        layers.append(
            GammaLayerFit(
                name=name,
                size=score.numel(),
                positive_count=positive_count,
                shape=shape,
                scale=scale,
                fallback_reason=reason,
            )
        )
    return GammaAllocationProblem(tuple(layers), sum(layer.size for layer in layers))


def solve_gamma_mom_rates(
    problem: GammaAllocationProblem,
    target_ratio: float,
    *,
    xtol: float = 1e-10,
    max_iterations: int = 200,
    max_bracket_expansions: int = 64,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Solve the shared Gamma-CDF threshold and return rates plus diagnostics."""
    if not math.isfinite(target_ratio) or not 0.0 <= target_ratio <= 1.0:
        raise ValueError(f"target_ratio must be in [0, 1], got {target_ratio}")
    if not math.isfinite(xtol) or xtol <= 0:
        raise ValueError(f"xtol must be positive, got {xtol}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}")
    if max_bracket_expansions < 0:
        raise ValueError(
            f"max_bracket_expansions must be non-negative, got {max_bracket_expansions}"
        )

    uniform_rates = {layer.name: target_ratio for layer in problem.layers}
    total = problem.total_parameters
    fitted = [layer for layer in problem.layers if layer.valid]
    fallback_layers = [layer for layer in problem.layers if not layer.valid]

    def metadata(
        status: str,
        fallback: Optional[str],
        threshold: Optional[float],
        calls: int,
        bounds,
        rates: Dict[str, float],
    ) -> Dict[str, object]:
        weighted_rate = (
            sum(layer.size * rates[layer.name] for layer in problem.layers) / total
            if total
            else 0.0
        )
        return {
            "status": status,
            "fallback": fallback,
            "threshold": threshold,
            "iterations": max(calls - 2, 0),
            "function_calls": calls,
            "target_ratio": target_ratio,
            "total_parameters": total,
            "fitted_layers": len(fitted),
            "fallback_layers": len(fallback_layers),
            "weighted_rate": weighted_rate,
            "budget_residual": weighted_rate - target_ratio if total else 0.0,
            "search_bounds": list(bounds) if bounds is not None else None,
        }

    if total == 0:
        return {}, metadata("empty", None, None, 0, None, {})
    if target_ratio in (0.0, 1.0):
        return uniform_rates, metadata("boundary", None, None, 0, None, uniform_rates)
    if not fitted:
        return uniform_rates, metadata(
            "all_invalid", "all_fits_invalid", None, 0, None, uniform_rates
        )

    calls = 0

    def objective(threshold: float) -> float:
        nonlocal calls
        calls += 1
        expected = 0.0
        for layer in problem.layers:
            ratio = target_ratio
            if layer.valid:
                shape, scale = layer.parameters()
                ratio = stats.gamma.cdf(threshold, shape, scale=scale)
            expected += layer.size * ratio
        value = expected / total - target_ratio
        return value if math.isfinite(value) else math.nan

    lower = 0.0
    upper = max(
        max(shape * scale for shape, scale in (layer.parameters() for layer in fitted)), 1.0
    )
    bracket_found = False
    for _ in range(max_bracket_expansions + 1):
        value = objective(upper)
        if math.isfinite(value) and value >= 0:
            bracket_found = True
            break
        upper *= 2.0

    if not bracket_found:
        return uniform_rates, metadata(
            "uniform_fallback",
            "root_not_bracketed",
            None,
            calls,
            (lower, upper),
            uniform_rates,
        )

    try:
        threshold = bisect(
            objective,
            lower,
            upper,
            xtol=xtol,
            maxiter=max_iterations,
        )
    except (ValueError, RuntimeError):
        return uniform_rates, metadata(
            "uniform_fallback",
            "bisection_failed",
            None,
            calls,
            (lower, upper),
            uniform_rates,
        )

    rates = {}
    for layer in problem.layers:
        ratio = target_ratio
        if layer.valid:
            shape, scale = layer.parameters()
            ratio = stats.gamma.cdf(threshold, shape, scale=scale)
        rates[layer.name] = max(0.0, min(1.0, float(ratio)))
    return rates, metadata(
        "converged",
        None,
        float(threshold),
        calls,
        (lower, upper),
        rates,
    )
