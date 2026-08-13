"""Demonstrate actual checkpoint compression with bitmask serialization.

Saves compressed checkpoints at various sparsity ratios and reports
actual file sizes vs original checkpoint size.

Usage:
    python experiments/scripts/run_checkpoint_size_demo.py \
        --model gpt2-medium \
        --dataset wikitext103 \
        --checkpoint /path/to/checkpoint.pt \
        --prune_ratios 0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 \
        --alpha 0.7 \
        --output_dir results/paper_results/checkpoint_sizes
"""

import argparse
import copy
import math
import os

os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dacp.pruning import apply_pruning
from dacp.pruning.allocation import get_allocation_strategy
from dacp.pruning.importance import combine_scores_2d_with_protection
from dacp.tools.checkpoint_io import (
    get_size_breakdown,
    load_compressed_checkpoint,
    save_compressed_checkpoint,
)
from experiments.lib.data import cache_batches, get_data_loaders
from experiments.lib.evaluation import evaluate as eval_fn
from experiments.lib.importance_compare.scoring import compute_scores_by_method
from experiments.lib.models import load_model
from experiments.lib.residual_runtime import write_json

METHOD_CONFIGS = {
    "magnitude": {
        "label": "Magnitude+Uniform",
        "scores_key": "magnitude",
        "allocation": "uniform",
    },
    "ours_2d": {
        "label": "Ours (2D)",
        "scores_key": "_2d_combined",
        "allocation": "gamma-adaptive",
        "alloc_scores_key": "second-order-hvp",
    },
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def _parse_prune_ratios(value: str) -> list[float]:
    try:
        ratios = [_finite_float(item.strip()) for item in value.split(",") if item.strip()]
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("prune ratios must be comma-separated numbers") from error
    if not ratios or any(not 0.0 < ratio < 1.0 for ratio in ratios):
        raise argparse.ArgumentTypeError("prune ratios must be in (0, 1)")
    return ratios


def _parse_methods(value: str) -> list[str]:
    methods = [method.strip() for method in value.split(",") if method.strip()]
    unknown = sorted(set(methods) - set(METHOD_CONFIGS))
    if not methods or unknown:
        raise argparse.ArgumentTypeError(
            f"methods must be selected from {sorted(METHOD_CONFIGS)}; unknown={unknown}"
        )
    return methods


def _quality_metric(metrics: dict[str, float], task_type: str) -> tuple[str, float, bool]:
    contracts = {
        "lm": ("perplexity", False),
        "cls": ("accuracy", True),
        "cv": ("accuracy", True),
        "reg": ("pearson", True),
    }
    if task_type not in contracts:
        raise ValueError(f"Unsupported task type for checkpoint-size evidence: {task_type}")
    name, higher_is_better = contracts[task_type]
    if name not in metrics:
        raise KeyError(f"{task_type} evaluation metrics must contain {name!r}")
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return name, value, higher_is_better


def _relative_quality_drop(value: float, baseline: float, *, higher_is_better: bool) -> float:
    if baseline == 0.0:
        if value == 0.0:
            return 0.0
        raise ValueError("relative quality drop is undefined for a zero baseline")
    delta = baseline - value if higher_is_better else value - baseline
    return delta / abs(baseline) * 100.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint size demonstration")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--prune_ratios",
        type=_parse_prune_ratios,
        default=_parse_prune_ratios("0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5"),
    )
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=_parse_methods("magnitude,ours_2d"),
        help="Comma-separated: magnitude, ours_2d",
    )
    parser.add_argument("--alpha", type=_finite_float, default=0.7)
    parser.add_argument("--hvp_batches", type=_positive_int, default=8)
    parser.add_argument("--eval_batches", type=_positive_int, default=50)
    parser.add_argument("--batch_size", type=_positive_int, default=2)
    parser.add_argument("--seq_length", type=_positive_int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/paper_results/checkpoint_sizes")
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="Actually save compressed .pt files (otherwise just compute sizes)",
    )
    return parser


def format_bytes(b):
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def main():
    args = _build_parser().parse_args()
    prune_ratios = args.prune_ratios
    method_names = args.methods

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Checkpoint Size Demonstration")
    print(f"Model: {args.model} | Dataset: {args.dataset}")
    print(f"Ratios: {prune_ratios}")
    print(f"Methods: {method_names}")
    print("=" * 70)

    # Load model
    model_init, model_family = load_model(
        args.model,
        pretrained=True,
        checkpoint_path=args.checkpoint,
        device="cpu",
        dataset_name=args.dataset,
    )

    # Load data
    train_loader, val_loader, task_type = get_data_loaders(
        args.model, args.dataset, args.batch_size, args.seq_length
    )

    # Cache data
    print("\nCaching data...")
    cached_eval = cache_batches(val_loader, args.eval_batches, task_type)
    cached_train = cache_batches(train_loader, args.hvp_batches + 2, task_type)

    # Original checkpoint size
    orig_path = args.checkpoint
    if orig_path and os.path.exists(orig_path):
        orig_size = os.path.getsize(orig_path)
    else:
        # Estimate from state_dict
        orig_size = sum(p.numel() * p.element_size() for p in model_init.parameters())
    print(f"\nOriginal checkpoint: {format_bytes(orig_size)}")

    # Baseline evaluation
    print("\nBaseline evaluation...")
    model_base = copy.deepcopy(model_init).to(args.device)
    baseline = eval_fn(model_base, cached_eval, task_type, args.device)
    metric_name, baseline_quality, higher_is_better = _quality_metric(baseline, task_type)
    print(f"  Baseline: loss={baseline['loss']:.4f}, " f"{metric_name}={baseline_quality:.4f}")
    del model_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute importance scores
    print("\nComputing importance scores...")
    all_methods = ["magnitude"]
    if "ours_2d" in method_names:
        all_methods.extend(["second-order-hvp", "first-order"])

    model_s = copy.deepcopy(model_init).to(args.device)
    score_cache = compute_scores_by_method(
        model_s,
        cached_train,
        task_type,
        methods=list(set(all_methods)),
        alpha=args.alpha,
        hvp_batches=args.hvp_batches,
        model_family=model_family,
    )
    del model_s
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build 2D combined scores
    if "ours_2d" in method_names:
        hvp_scores = score_cache.get("second-order-hvp")
        mag_scores = score_cache.get("magnitude")
        if hvp_scores and mag_scores:
            score_cache["_2d_combined"] = combine_scores_2d_with_protection(
                mag_scores, hvp_scores, protection_ratio=0.001, alpha=args.alpha
            )
            print("  2D combined scores computed")

    results = []

    print("\n" + "=" * 70)
    print(
        f"{'Method':<25} {'Sparsity':>8} {metric_name:>11} {'Orig':>10} "
        f"{'Compressed':>10} {'Ratio':>6} {'Saved':>6}"
    )
    print("-" * 70)

    for method_name in method_names:
        cfg = METHOD_CONFIGS[method_name]
        scores = score_cache[cfg["scores_key"]]
        alloc_scores = score_cache.get(cfg.get("alloc_scores_key", cfg["scores_key"]), scores)
        allocator = get_allocation_strategy(cfg["allocation"])

        for ratio in prune_ratios:
            # Apply pruning
            model = copy.deepcopy(model_init).to(args.device)
            layer_ratios = allocator.allocate(alloc_scores, ratio)
            _, masks, actual_ratio = apply_pruning(model, scores, layer_ratios, device=args.device)

            # Evaluate
            metrics = eval_fn(model, cached_eval, task_type, args.device)

            # Get original state_dict (before pruning) for size computation
            orig_sd = {n: p.detach().cpu() for n, p in model_init.named_parameters()}

            # Compute actual compressed size
            breakdown = get_size_breakdown(orig_sd, masks, use_fp16=True)

            # Optionally save actual file
            file_size = breakdown["total_compressed_bytes"]
            if args.save_checkpoints:
                fname = f"{cfg['label'].replace(' ', '_')}_{int(ratio*100)}pct.pt"
                fpath = output_dir / fname
                file_size = save_compressed_checkpoint(orig_sd, masks, str(fpath))

                # Verify: load back and check
                restored = load_compressed_checkpoint(str(fpath))
                if set(restored) != set(orig_sd):
                    raise RuntimeError("compressed checkpoint round-trip changed parameter keys")

            result_metric_name, quality, _ = _quality_metric(metrics, task_type)
            if result_metric_name != metric_name:
                raise RuntimeError("evaluation metric contract changed during the experiment")
            saved_pct = (1 - file_size / orig_size) * 100
            print(
                f"{cfg['label']:<25} {ratio:>7.0%} {quality:>11.4f} "
                f"{format_bytes(orig_size):>10} {format_bytes(file_size):>10} "
                f"{orig_size/file_size:>5.2f}x {saved_pct:>5.1f}%"
            )

            results.append(
                {
                    "method": cfg["label"],
                    "target_sparsity": ratio,
                    "actual_sparsity": actual_ratio,
                    "loss": metrics["loss"],
                    "metric": metric_name,
                    "metric_value": quality,
                    "original_bytes": orig_size,
                    "compressed_bytes": file_size,
                    "theoretical_bytes": breakdown["total_compressed_bytes"],
                    "mask_bytes": breakdown["mask_bytes"],
                    "values_bytes": breakdown["values_bytes"],
                    "compression_ratio": orig_size / file_size,
                    "storage_saved_pct": saved_pct,
                }
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("=" * 70)

    # Save results
    results_path = output_dir / f"checkpoint_sizes_{args.model}_{args.dataset}.json"
    write_json(
        results_path,
        {
            "config": vars(args),
            "baseline": baseline,
            "original_size_bytes": orig_size,
            "results": results,
        },
    )
    print(f"\nResults saved to {results_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Quality vs Compression Trade-off")
    print("-" * 70)
    for r in results:
        degradation = _relative_quality_drop(
            r["metric_value"], baseline_quality, higher_is_better=higher_is_better
        )
        print(
            f"  {r['method']:<25} | {r['target_sparsity']:>5.0%} sparsity | "
            f"{metric_name}={r['metric_value']:>7.4f} ({degradation:>+5.1f}%) | "
            f"Size: {format_bytes(r['compressed_bytes']):>8} ({r['compression_ratio']:.2f}x)"
        )


if __name__ == "__main__":
    main()
