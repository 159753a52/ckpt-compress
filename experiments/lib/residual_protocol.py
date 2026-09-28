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
TAYLOR_SIGNED_BENEFIT_WEIBULL_METHOD = "taylor_signed_benefit_weibull"
TAYLOR_SIGNED_FIRST_ORDER_UNIFORM_METHOD = "taylor_signed_fo_uniform"
TAYLOR_SIGNED_SECOND_ORDER_UNIFORM_METHOD = "taylor_signed_so_uniform"
TAYLOR_MEANABS_UNIFORM_METHOD = "taylor_meanabs_uniform"
TAYLOR_MEANABS_WEIBULL_MOM_METHOD = "taylor_meanabs_weibull_mom"
TAYLOR_COMPONENTS_UNIFORM_METHOD = "taylor_components_uniform"
TAYLOR_COMPONENTS_EXACT_GLOBAL_METHOD = "taylor_components_exact_global"

# Residual first-order score |g*delta| with exact global selection. Distinct
# from the full-weight ``first_order_exact_global`` internal method used by the
# Inshrinkerator-style adapter.
RESIDUAL_FIRST_ORDER_EXACT_GLOBAL_METHOD = "residual_first_order_exact_global"

# Coverage-tracked Gauss--Newton probe scores |-g*d + 0.5*d*(F_hat d)| with
# exact global selection; ids are ``probe_<estimator>_m<rounds>_exact_global``.
PROBE_ESTIMATORS = ("sample", "topk", "headtail")
PROBE_ROUNDS = (1, 2, 4, 8)

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


def probe_method_id(estimator: str, rounds: int) -> str:
    if estimator not in PROBE_ESTIMATORS:
        raise ValueError(f"Unknown probe estimator: {estimator!r}")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError(f"Probe rounds must be a positive integer, got {rounds!r}")
    return f"probe_{estimator}_m{rounds}_exact_global"


PROBE_METHODS = {
    probe_method_id(estimator, rounds): (estimator, rounds)
    for estimator in PROBE_ESTIMATORS
    for rounds in PROBE_ROUNDS
}


def parse_probe_method_id(method: object) -> tuple[str, int] | None:
    """Return ``(estimator, rounds)`` for a registered probe method, else None."""
    return PROBE_METHODS.get(method) if isinstance(method, str) else None


__all__ = [
    "BASE_METHODS",
    "FIRST_ORDER_UNIFORM_METHOD",
    "FIXED_METHODS",
    "NO_COMPRESSION_METHOD",
    "PROBE_ESTIMATORS",
    "PROBE_METHODS",
    "PROBE_ROUNDS",
    "RESIDUAL_FIRST_ORDER_EXACT_GLOBAL_METHOD",
    "parse_probe_method_id",
    "probe_method_id",
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
