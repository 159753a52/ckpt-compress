"""Shared method-allocation configuration for residual experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AdaptiveMethodConfig:
    """Allocation settings shared by one assembled method family."""

    prune_ratio: float
    max_layer_ratio: float
    probe_radius: float
    trust_radii: Tuple[float, ...]
    spectral_ranks: Tuple[int, ...]
    spectral_probe_radius: float
    spectral_trust_radius: float
    quantile_smoothness_values: Tuple[float, ...]
    quantile_trust_radius: float
    quantile_cost_normalization: str
    device: str


__all__ = ["AdaptiveMethodConfig"]
