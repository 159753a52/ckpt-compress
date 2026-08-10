"""Shared scalar validation for pruning configuration."""

import math


def validate_unit_interval(name: str, value: float) -> float:
    """Normalize a finite numeric value constrained to ``[0, 1]``."""
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a finite number in [0, 1], got {value}"
        )
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite number in [0, 1], got {value}"
        ) from exc
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(
            f"{name} must be a finite number in [0, 1], got {value}"
        )
    return normalized


__all__ = ["validate_unit_interval"]
