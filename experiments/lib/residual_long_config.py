"""Typed configuration for the long residual recovery experiment."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from experiments.lib.residual_method_config import AdaptiveMethodConfig


@dataclass(frozen=True)
class LongExperimentConfig:
    """Normalized inputs for one complete long residual experiment."""

    checkpoint: Path
    data_dir: Path
    output_dir: Path
    seeds: Tuple[int, ...]
    total_steps: int
    recovery_step: int
    batch_size: int
    seq_length: int
    hvp_batches: int
    allocation_probe_batches: int
    allocation_selection_batches: int
    eval_batches: int
    train_pool_batches: int
    learning_rate: float
    methods: AdaptiveMethodConfig

    @property
    def device(self) -> str:
        return self.methods.device

    @property
    def continuation_steps(self) -> int:
        return self.total_steps - self.recovery_step

    def to_result_dict(self) -> Dict[str, object]:
        """Return the existing flat, JSON-ready experiment config schema."""
        methods = self.methods
        return {
            "checkpoint": str(self.checkpoint.resolve()),
            "data_dir": str(self.data_dir.resolve()),
            "seeds": list(self.seeds),
            "total_steps": self.total_steps,
            "recovery_step": self.recovery_step,
            "prune_ratio": methods.prune_ratio,
            "max_layer_ratio": methods.max_layer_ratio,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "hvp_batches": self.hvp_batches,
            "allocation_probe_batches": self.allocation_probe_batches,
            "allocation_selection_batches": self.allocation_selection_batches,
            "allocation_probe_radius": methods.probe_radius,
            "allocation_trust_radii": list(methods.trust_radii),
            "spectral_ranks": list(methods.spectral_ranks),
            "spectral_probe_radius": methods.spectral_probe_radius,
            "spectral_trust_radius": methods.spectral_trust_radius,
            "quantile_smoothness_values": list(methods.quantile_smoothness_values),
            "quantile_trust_radius": methods.quantile_trust_radius,
            "quantile_cost_normalization": methods.quantile_cost_normalization,
            "eval_batches": self.eval_batches,
            "train_pool_batches": self.train_pool_batches,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "output_dir": str(self.output_dir.resolve()),
            "scheduler": "cosine",
            "continuation_steps": self.continuation_steps,
        }


def normalize_long_config(args: argparse.Namespace) -> LongExperimentConfig:
    """Normalize CLI strings without changing their established ordering rules."""
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    trust_radii = tuple(
        float(value)
        for value in args.allocation_trust_radii.split(",")
        if value.strip()
    )
    spectral_ranks = tuple(
        sorted({int(value) for value in args.spectral_ranks.split(",") if value.strip()})
    )
    quantile_smoothness_values = tuple(
        sorted(
            {
                float(value)
                for value in args.quantile_smoothness_values.split(",")
                if value.strip()
            }
        )
    )
    methods = AdaptiveMethodConfig(
        prune_ratio=args.prune_ratio,
        max_layer_ratio=args.max_layer_ratio,
        probe_radius=args.allocation_probe_radius,
        trust_radii=trust_radii,
        spectral_ranks=spectral_ranks,
        spectral_probe_radius=args.spectral_probe_radius,
        spectral_trust_radius=args.spectral_trust_radius,
        quantile_smoothness_values=quantile_smoothness_values,
        quantile_trust_radius=args.quantile_trust_radius,
        quantile_cost_normalization=args.quantile_cost_normalization,
        device=args.device,
    )
    return LongExperimentConfig(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seeds=seeds,
        total_steps=args.total_steps,
        recovery_step=args.recovery_step,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        hvp_batches=args.hvp_batches,
        allocation_probe_batches=args.allocation_probe_batches,
        allocation_selection_batches=args.allocation_selection_batches,
        eval_batches=args.eval_batches,
        train_pool_batches=args.train_pool_batches,
        learning_rate=args.learning_rate,
        methods=methods,
    )


def validate_long_config(config: LongExperimentConfig) -> None:
    """Validate normalized values in the legacy error-priority order."""
    methods = config.methods
    if not methods.trust_radii or any(value <= 0 for value in methods.trust_radii):
        raise ValueError("--allocation-trust-radii must contain positive values")
    if any(value > 1 for value in methods.trust_radii):
        raise ValueError("--allocation-trust-radii values cannot exceed 1")
    if any(rank < 1 for rank in methods.spectral_ranks):
        raise ValueError("--spectral-ranks must contain positive integers")
    if methods.quantile_smoothness_values and any(
        not math.isfinite(value) or value <= 0
        for value in methods.quantile_smoothness_values
    ):
        raise ValueError("--quantile-smoothness-values must contain positive values")
    if methods.quantile_smoothness_values and not (
        0
        < methods.quantile_trust_radius
        <= min(
            methods.prune_ratio,
            methods.max_layer_ratio - methods.prune_ratio,
        )
    ):
        raise ValueError("--quantile-trust-radius must fit inside the layer-rate box")
    if not config.seeds:
        raise ValueError("At least one seed is required")
    if not 0 < config.recovery_step < config.total_steps:
        raise ValueError("--recovery-step must be strictly inside --total-steps")
    if not 0 < methods.prune_ratio < 1:
        raise ValueError("--prune-ratio must be in (0, 1)")
    if not methods.prune_ratio <= methods.max_layer_ratio <= 1:
        raise ValueError("--max-layer-ratio must be in [prune_ratio, 1]")
    if config.allocation_probe_batches < 1 or config.allocation_selection_batches < 1:
        raise ValueError("Allocation probe and selection batch counts must be positive")
    if not 0 < methods.probe_radius < min(
        methods.prune_ratio,
        1.0 - methods.prune_ratio,
    ):
        raise ValueError("--allocation-probe-radius must fit around --prune-ratio")
    if methods.prune_ratio + methods.probe_radius > methods.max_layer_ratio:
        raise ValueError("--max-layer-ratio must leave room for the positive allocation probe")
    if not 0 < methods.spectral_probe_radius < min(
        methods.prune_ratio,
        1.0 - methods.prune_ratio,
    ):
        raise ValueError("--spectral-probe-radius must fit around --prune-ratio")
    if methods.prune_ratio + methods.spectral_probe_radius > methods.max_layer_ratio:
        raise ValueError("--max-layer-ratio must leave room for the positive spectral probe")
    if not 0 < methods.spectral_trust_radius <= 1:
        raise ValueError("--spectral-trust-radius must be in (0, 1]")
    needed = (
        config.total_steps
        + config.hvp_batches
        + config.allocation_probe_batches
        + config.allocation_selection_batches
    )
    if config.train_pool_batches < needed:
        raise ValueError(f"--train-pool-batches must be at least {needed}")


__all__ = [
    "AdaptiveMethodConfig",
    "LongExperimentConfig",
    "normalize_long_config",
    "validate_long_config",
]
