"""Stable method identifiers for residual recovery results and comparisons."""

from __future__ import annotations

RESIDUAL_MAGNITUDE_UNIFORM_METHOD = "residual_magnitude_uniform"
RESIDUAL_MAGNITUDE_WEIBULL_MOM_METHOD = "residual_magnitude_weibull_mom"
FIRST_ORDER_UNIFORM_METHOD = "first_order_uniform"
SECOND_ORDER_UNIFORM_METHOD = "second_order_uniform"
TAYLOR_UNIFORM_METHOD = "taylor_uniform"
TAYLOR_WEIBULL_MOM_METHOD = "taylor_weibull_mom"
TAYLOR_EXACT_GLOBAL_METHOD = "taylor_exact_global"
TAYLOR_PROBE_TRUST_METHOD = "taylor_probe_trust"
NO_COMPRESSION_METHOD = "no_compression"

# Optimized-implementation score-aggregation variants (diagnostic track).
# `signed` keeps the per-coordinate Taylor contribution's sign instead of
# taking an absolute value, so ascending order prunes coordinates whose
# reversion is predicted to reduce loss first.
TAYLOR_SIGNED_UNIFORM_METHOD = "taylor_signed_uniform"
TAYLOR_SIGNED_EXACT_GLOBAL_METHOD = "taylor_signed_exact_global"
TAYLOR_SIGNED_WEIBULL_MOM_METHOD = "taylor_signed_weibull_mom"
TAYLOR_SIGNED_FIRST_ORDER_UNIFORM_METHOD = "taylor_signed_fo_uniform"
TAYLOR_SIGNED_SECOND_ORDER_UNIFORM_METHOD = "taylor_signed_so_uniform"
TAYLOR_MEANABS_UNIFORM_METHOD = "taylor_meanabs_uniform"
TAYLOR_MEANABS_WEIBULL_MOM_METHOD = "taylor_meanabs_weibull_mom"
TAYLOR_COMPONENTS_UNIFORM_METHOD = "taylor_components_uniform"
TAYLOR_COMPONENTS_EXACT_GLOBAL_METHOD = "taylor_components_exact_global"

TAYLOR_METHOD_PREFIX = "taylor_"
SPECTRAL_METHOD_PREFIX = "taylor_spectral_k"
QUANTILE_METHOD_PREFIX = "taylor_quantile_"

BASE_METHODS = (
    RESIDUAL_MAGNITUDE_UNIFORM_METHOD,
    FIRST_ORDER_UNIFORM_METHOD,
    SECOND_ORDER_UNIFORM_METHOD,
    TAYLOR_UNIFORM_METHOD,
    TAYLOR_WEIBULL_MOM_METHOD,
    TAYLOR_EXACT_GLOBAL_METHOD,
)
SHORT_GATE_METHODS = (
    RESIDUAL_MAGNITUDE_UNIFORM_METHOD,
    TAYLOR_UNIFORM_METHOD,
    TAYLOR_WEIBULL_MOM_METHOD,
    TAYLOR_EXACT_GLOBAL_METHOD,
)
FIXED_METHODS = (*BASE_METHODS, TAYLOR_PROBE_TRUST_METHOD)


def float_slug(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p").replace("+", "")


def spectral_method_id(rank: int) -> str:
    return f"{SPECTRAL_METHOD_PREFIX}{rank}"


def quantile_method_id(cost_normalization: str, smoothness: float) -> str:
    normalization_slug = "relative" if cost_normalization == "layer_uniform_cost" else "global"
    return f"{QUANTILE_METHOD_PREFIX}{normalization_slug}_smooth_l{float_slug(smoothness)}"


__all__ = [
    "BASE_METHODS",
    "FIRST_ORDER_UNIFORM_METHOD",
    "FIXED_METHODS",
    "NO_COMPRESSION_METHOD",
    "QUANTILE_METHOD_PREFIX",
    "RESIDUAL_MAGNITUDE_UNIFORM_METHOD",
    "RESIDUAL_MAGNITUDE_WEIBULL_MOM_METHOD",
    "SECOND_ORDER_UNIFORM_METHOD",
    "SHORT_GATE_METHODS",
    "SPECTRAL_METHOD_PREFIX",
    "TAYLOR_COMPONENTS_EXACT_GLOBAL_METHOD",
    "TAYLOR_COMPONENTS_UNIFORM_METHOD",
    "TAYLOR_EXACT_GLOBAL_METHOD",
    "TAYLOR_MEANABS_UNIFORM_METHOD",
    "TAYLOR_MEANABS_WEIBULL_MOM_METHOD",
    "TAYLOR_METHOD_PREFIX",
    "TAYLOR_PROBE_TRUST_METHOD",
    "TAYLOR_SIGNED_EXACT_GLOBAL_METHOD",
    "TAYLOR_SIGNED_FIRST_ORDER_UNIFORM_METHOD",
    "TAYLOR_SIGNED_SECOND_ORDER_UNIFORM_METHOD",
    "TAYLOR_SIGNED_UNIFORM_METHOD",
    "TAYLOR_SIGNED_WEIBULL_MOM_METHOD",
    "TAYLOR_UNIFORM_METHOD",
    "TAYLOR_WEIBULL_MOM_METHOD",
    "float_slug",
    "quantile_method_id",
    "spectral_method_id",
]
