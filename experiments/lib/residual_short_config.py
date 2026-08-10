"""CLI configuration for the short residual allocation gate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]

SHORT_GATE_DESCRIPTION = """Reusable residual checkpoint recovery and allocation diagnostics.

This diagnostic computes one block-diagonal HVP at the current checkpoint and
reuses the resulting residual Taylor scores for four masks:

* residual magnitude + uniform per-layer allocation
* residual Taylor score + uniform per-layer allocation
* residual Taylor score + two-parameter Weibull MoM allocation
* residual Taylor score + exact empirical global threshold

By default no optimizer is constructed and no continuation training is
performed.  An optional, explicitly counted matched continuation can be run
after scoring.  A keep mask is applied to the checkpoint residual, not to the
full model weight.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the established short-gate CLI without normalizing its values."""
    parser = argparse.ArgumentParser(description=SHORT_GATE_DESCRIPTION)
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/gpt2_medium_wikitext103_1000steps/checkpoint_step_800.pt",
    )
    parser.add_argument(
        "--current-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/gpt2_medium_wikitext103_1000steps/checkpoint_step_1000.pt",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT.parent.parent / "data/wikitext103",
    )
    parser.add_argument("--prune-ratio", type=float, default=0.30)
    parser.add_argument("--max-layer-ratio", type=float, default=0.80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--hvp-batches", type=int, default=1)
    parser.add_argument(
        "--train-batch-offset",
        type=int,
        default=0,
        help="Skip this many deterministic training batches before HVP data.",
    )
    parser.add_argument("--continuation-steps", type=int, default=0)
    parser.add_argument("--continuation-lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/diagnostics/v100_allocation_gate",
    )
    return parser.parse_args(argv)


def validate_short_config(args: argparse.Namespace) -> None:
    """Validate scalar values in the short gate's established priority order."""
    if not 0 < args.prune_ratio < 1:
        raise ValueError("--prune-ratio must be in (0, 1)")
    if args.hvp_batches < 1 or args.continuation_steps < 0 or args.train_batch_offset < 0:
        raise ValueError("HVP batches must be positive and step/offset counts non-negative")


__all__ = ["ROOT", "SHORT_GATE_DESCRIPTION", "parse_args", "validate_short_config"]
