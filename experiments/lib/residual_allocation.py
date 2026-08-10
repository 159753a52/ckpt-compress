"""Compatibility facade for residual allocation algorithms."""

from __future__ import annotations

from experiments.lib.residual_budget import (
    bounded_largest_remainder_counts,
    largest_remainder_counts,
    trust_region_counts,
    uniform_counts,
)
from experiments.lib.residual_quantile import calibrate_quantile_smooth_allocation
from experiments.lib.residual_spectral import (
    budget_tangent_dct_directions,
    directional_layer_counts,
    reconstruct_directional_gradient,
)
from experiments.lib.residual_weibull import (
    fit_weibull_mom,
    weibull_cdf,
    weibull_counts,
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
