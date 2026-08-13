#!/usr/bin/env python3
"""Print validated single-shot and fault-tolerant paper results."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import yaml

ROOT = Path(__file__).parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from experiments.lib.result_schema import load_result_bundle
from experiments.scripts.aggregate_results import (
    MetricEntry,
    ResultScanError,
    extract_final_metrics,
)


def _metric_text(entry: MetricEntry) -> str:
    metric = entry["metric"]
    value = entry["value"]
    if metric == "perplexity":
        return f"PPL={value:.2f}"
    if metric in ("accuracy", "top1_accuracy", "pearson"):
        return f"{metric}={value:.4f}"
    return f"{metric}={value:.4f}"


def _print_files(paths: Sequence[Path], *, lenient: bool) -> int:
    printed = 0
    for path in paths:
        try:
            entries = extract_final_metrics(load_result_bundle(path))
        except (OSError, TypeError, ValueError, UnicodeError, yaml.YAMLError) as exc:
            if lenient:
                print(f"warning: skipped {path}: {exc}", file=sys.stderr)
                continue
            raise ResultScanError(f"invalid result file {path}: {exc}") from exc

        print(f"\n--- {path.name} ---")
        for entry in entries:
            print(
                f"  {entry['method']:40s}  "
                f"sparsity={entry['prune_ratio']:.0%}  {_metric_text(entry)}"
            )
        printed += 1
    return printed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "paper_results",
        help="Paper-results directory",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Skip invalid matching result files during exploratory inspection",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    sections = (
        (
            "TABLE 1 (Single-shot compression)",
            args.results_dir / "table1",
            "*_table1_*.json",
        ),
        (
            "FAULT-TOLERANT TRAINING RESULTS",
            args.results_dir / "fault_tolerant",
            "*_ft_*.json",
        ),
    )

    total = 0
    try:
        for title, directory, pattern in sections:
            print("=" * 80)
            print(title)
            print("=" * 80)
            paths = sorted(
                path for path in directory.glob(pattern) if not path.name.endswith("_config.json")
            )
            total += _print_files(paths, lenient=args.lenient)
    except ResultScanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if total == 0:
        print(f"error: no matching valid results under {args.results_dir}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
