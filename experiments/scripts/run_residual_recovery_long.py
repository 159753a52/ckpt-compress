"""Run a longer, paired residual checkpoint recovery experiment on one V100."""

from __future__ import annotations

import argparse
import copy
import gc
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.residual_allocation import (  # noqa: E402
    calibrate_quantile_smooth_allocation,
)
from experiments.lib.residual_calibration import (  # noqa: E402
    calibrate_spectral_allocation,
    calibrate_trust_region_allocation,
)
from experiments.lib.residual_masks import (  # noqa: E402
    MaskDict,
    layer_masks,
    layer_rates,
    layer_score_orders,
    mask_metrics,
    mask_overlap,
    restore_with_mask,
)
from experiments.lib.residual_methods import (  # noqa: E402
    build_masks,
    float_slug,
    quantile_method_id,
    score_for_method,
    spectral_method_id,
)
from experiments.lib.residual_protocol import (  # noqa: E402
    NO_COMPRESSION_METHOD,
    TAYLOR_EXACT_GLOBAL_METHOD,
    TAYLOR_PROBE_TRUST_METHOD,
)
from experiments.lib.residual_runtime import (  # noqa: E402
    batch_hash,
    configure_hf_offline,
    empty_device_cache,
    evaluate_lm,
    lm_loss,
    load_token_batches,
    load_training_checkpoint,
    optimizer_state_to_cpu,
    peak_memory_bytes,
    reset_peak_memory,
    sha256_file,
    write_json,
)
from experiments.lib.residual_reporting import (  # noqa: E402
    aggregate,
    summarize_values,
)
from experiments.lib.residual_scoring import (  # noqa: E402
    compute_block_taylor_scores,
    eligible_layers,
)
from experiments.lib.residual_training import (  # noqa: E402
    build_optimizer,
    clone_model_state_to_cpu,
    partition_seed_batches,
    seeded_training_batches,
    train_segment,
)


@dataclass(frozen=True)
class LongRunContext:
    """Resources shared across every paired seed in one long experiment."""

    args: argparse.Namespace
    output_path: Path
    results: Dict[str, object]
    model_factory: Callable[[], torch.nn.Module]
    training_pool: Sequence[Mapping[str, torch.Tensor]]
    eval_batches: Sequence[Mapping[str, torch.Tensor]]
    reference_state: Mapping[str, torch.Tensor]
    reference_optimizer_state: Mapping
    trust_radii: Sequence[float]
    spectral_ranks: Sequence[int]
    quantile_smoothness_values: Sequence[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/gpt2_medium_wikitext103_1000steps/checkpoint_step_1000.pt",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT.parent.parent / "data/wikitext103"
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--recovery-step", type=int, default=50)
    parser.add_argument("--prune-ratio", type=float, default=0.50)
    parser.add_argument("--max-layer-ratio", type=float, default=0.80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--hvp-batches", type=int, default=8)
    parser.add_argument("--allocation-probe-batches", type=int, default=8)
    parser.add_argument("--allocation-selection-batches", type=int, default=8)
    parser.add_argument("--allocation-probe-radius", type=float, default=0.025)
    parser.add_argument("--allocation-trust-radii", default="0.025,0.05,0.1")
    parser.add_argument(
        "--spectral-ranks",
        default="",
        help="Comma-separated DCT ranks; empty disables spectral allocation",
    )
    parser.add_argument("--spectral-probe-radius", type=float, default=0.025)
    parser.add_argument("--spectral-trust-radius", type=float, default=0.1)
    parser.add_argument(
        "--quantile-smoothness-values",
        default="",
        help="Comma-separated positive lambda values; empty disables the method",
    )
    parser.add_argument("--quantile-trust-radius", type=float, default=0.1)
    parser.add_argument(
        "--quantile-cost-normalization",
        choices=("global", "layer_uniform_cost"),
        default="global",
    )
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--train-pool-batches", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/diagnostics/residual_recovery_long",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, seeds: Sequence[int]) -> None:
    if not seeds:
        raise ValueError("At least one seed is required")
    if not 0 < args.recovery_step < args.total_steps:
        raise ValueError("--recovery-step must be strictly inside --total-steps")
    if not 0 < args.prune_ratio < 1:
        raise ValueError("--prune-ratio must be in (0, 1)")
    if not args.prune_ratio <= args.max_layer_ratio <= 1:
        raise ValueError("--max-layer-ratio must be in [prune_ratio, 1]")
    if args.allocation_probe_batches < 1 or args.allocation_selection_batches < 1:
        raise ValueError("Allocation probe and selection batch counts must be positive")
    if not 0 < args.allocation_probe_radius < min(args.prune_ratio, 1.0 - args.prune_ratio):
        raise ValueError("--allocation-probe-radius must fit around --prune-ratio")
    if args.prune_ratio + args.allocation_probe_radius > args.max_layer_ratio:
        raise ValueError("--max-layer-ratio must leave room for the positive allocation probe")
    if not 0 < args.spectral_probe_radius < min(args.prune_ratio, 1.0 - args.prune_ratio):
        raise ValueError("--spectral-probe-radius must fit around --prune-ratio")
    if args.prune_ratio + args.spectral_probe_radius > args.max_layer_ratio:
        raise ValueError("--max-layer-ratio must leave room for the positive spectral probe")
    if not 0 < args.spectral_trust_radius <= 1:
        raise ValueError("--spectral-trust-radius must be in (0, 1]")
    needed = (
        args.total_steps
        + args.hvp_batches
        + args.allocation_probe_batches
        + args.allocation_selection_batches
    )
    if args.train_pool_batches < needed:
        raise ValueError(f"--train-pool-batches must be at least {needed}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def run_seed(
    context: LongRunContext,
    seed: int,
    seed_index: int,
    seed_count: int,
) -> None:
    """Run one paired seed while preserving incremental result checkpoints."""
    args = context.args
    output_path = context.output_path
    results = context.results
    reference_state = context.reference_state
    eval_batches = context.eval_batches

    print(f"\n=== Seed {seed} ({seed_index}/{seed_count}) ===", flush=True)
    seed_started = time.perf_counter()
    batches = partition_seed_batches(
        context.training_pool,
        args.total_steps,
        args.recovery_step,
        args.hvp_batches,
        args.allocation_probe_batches,
        args.allocation_selection_batches,
        seed,
    )

    seed_result: Dict[str, object] = {
        "seed": seed,
        "selected_pool_indices": batches.selected_pool_indices,
        "data_hashes": batches.data_hashes(),
        "methods": {},
    }
    results["seed_results"][str(seed)] = seed_result
    write_json(output_path, results)

    model = context.model_factory()
    model.load_state_dict(reference_state, strict=True)
    model.to(args.device)
    reference_metrics = evaluate_lm(model, eval_batches, args.device)
    seed_result["reference"] = reference_metrics

    optimizer = build_optimizer(
        model,
        context.reference_optimizer_state,
        learning_rate=args.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=0.0
    )
    pre_metrics = train_segment(
        model, optimizer, scheduler, batches.pre_recovery, seed, args.device
    )
    current_state = clone_model_state_to_cpu(model)
    current_optimizer_state = optimizer_state_to_cpu(optimizer.state_dict())
    current_scheduler_state = copy.deepcopy(scheduler.state_dict())
    del optimizer, scheduler
    empty_device_cache(args.device)
    current_metrics = evaluate_lm(model, eval_batches, args.device)
    seed_result["pre_recovery_training"] = pre_metrics
    seed_result["current"] = current_metrics
    write_json(output_path, results)
    print(
        f"  reference PPL={reference_metrics['perplexity']:.4f}; "
        f"recovery-point PPL={current_metrics['perplexity']:.4f}",
        flush=True,
    )

    layers = eligible_layers(model)
    eligible_names = [name for layer in layers for name in layer]
    delta = {
        name: current_state[name].detach().float() - reference_state[name].detach().float()
        for name in eligible_names
    }
    magnitude_scores = {name: values.abs() for name, values in delta.items()}
    reset_peak_memory(args.device)
    components_raw, scoring_metrics = compute_block_taylor_scores(
        model,
        batches.scoring,
        layers,
        delta,
        args.device,
        return_components=True,
    )
    components = components_raw
    scoring_metrics["peak_gpu_memory_bytes"] = peak_memory_bytes(args.device)
    seed_result["scoring"] = scoring_metrics

    allocation_started = time.perf_counter()
    taylor_score_orders = None
    score_order_seconds = 0.0
    if context.spectral_ranks or context.quantile_smoothness_values:
        score_order_started = time.perf_counter()
        taylor_score_orders = layer_score_orders(
            layers, components["taylor"], sort_device=args.device
        )
        score_order_seconds = time.perf_counter() - score_order_started
    masks, allocation_metadata = build_masks(
        layers,
        magnitude_scores,
        components,
        args.prune_ratio,
        args.max_layer_ratio,
        taylor_score_orders,
    )
    probe_trust_counts, probe_trust_metadata = calibrate_trust_region_allocation(
        model,
        current_state,
        reference_state,
        layers,
        components["taylor"],
        allocation_metadata["uniform_layer_counts"],
        allocation_metadata["target_pruned"],
        args.prune_ratio,
        args.max_layer_ratio,
        batches.allocation_probe,
        batches.allocation_selection,
        args.allocation_probe_radius,
        context.trust_radii,
        args.device,
        taylor_score_orders,
    )
    masks[TAYLOR_PROBE_TRUST_METHOD] = layer_masks(
        layers, components["taylor"], probe_trust_counts, taylor_score_orders
    )
    allocation_metadata["probe_trust_layer_counts"] = probe_trust_counts
    allocation_metadata["probe_trust"] = probe_trust_metadata
    if context.spectral_ranks:
        spectral_counts, spectral_metadata = calibrate_spectral_allocation(
            model,
            current_state,
            reference_state,
            layers,
            components["taylor"],
            allocation_metadata["target_pruned"],
            args.prune_ratio,
            args.max_layer_ratio,
            batches.allocation_probe,
            args.spectral_probe_radius,
            context.spectral_ranks,
            args.spectral_trust_radius,
            args.device,
            taylor_score_orders,
        )
        for rank, counts in spectral_counts.items():
            masks[spectral_method_id(rank)] = layer_masks(
                layers, components["taylor"], counts, taylor_score_orders
            )
        allocation_metadata["spectral_layer_counts"] = spectral_counts
        allocation_metadata["spectral"] = spectral_metadata
    if context.quantile_smoothness_values:
        quantile_counts, quantile_metadata = calibrate_quantile_smooth_allocation(
            layers,
            components["taylor"],
            taylor_score_orders,
            allocation_metadata["target_pruned"],
            args.prune_ratio,
            args.max_layer_ratio,
            args.quantile_trust_radius,
            context.quantile_smoothness_values,
            args.device,
            args.quantile_cost_normalization,
        )
        for smoothness, counts in quantile_counts.items():
            method = quantile_method_id(
                args.quantile_cost_normalization,
                smoothness,
            )
            masks[method] = layer_masks(
                layers, components["taylor"], counts, taylor_score_orders
            )
        allocation_metadata["quantile_smooth_layer_counts"] = {
            str(value): counts for value, counts in quantile_counts.items()
        }
        allocation_metadata["quantile_smooth"] = quantile_metadata
    if taylor_score_orders is not None:
        allocation_metadata["score_order_seconds"] = score_order_seconds
        allocation_metadata["score_order_sort_device"] = args.device
        allocation_metadata["score_order_bytes"] = sum(
            order.numel() * order.element_size() for order in taylor_score_orders
        )
    del taylor_score_orders
    allocation_metadata["seconds"] = time.perf_counter() - allocation_started
    allocation_metadata["whole_model_parameters"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    seed_result["allocation"] = allocation_metadata
    exact_metrics = mask_metrics(
        masks[TAYLOR_EXACT_GLOBAL_METHOD], components["taylor"]
    )

    for method, method_masks in masks.items():
        metrics = mask_metrics(method_masks, components["taylor"])
        selection = mask_metrics(
            method_masks, score_for_method(method, magnitude_scores, components)
        )
        metrics["selection_score_cost"] = selection["proxy_cost"]
        metrics["eligible_sparsity"] = (
            metrics["pruned"] / allocation_metadata["eligible_parameters"]
        )
        metrics["layer_rates"] = layer_rates(layers, method_masks)
        metrics["taylor_regret_vs_exact"] = (
            (metrics["proxy_cost"] - exact_metrics["proxy_cost"])
            / max(abs(exact_metrics["proxy_cost"]), 1e-30)
        )
        metrics["overlap_with_taylor_exact"] = mask_overlap(
            method_masks, masks[TAYLOR_EXACT_GLOBAL_METHOD]
        )
        restore_with_mask(
            model, current_state, reference_state, method_masks, args.device
        )
        metrics["immediate"] = evaluate_lm(model, eval_batches, args.device)
        seed_result["methods"][method] = metrics
        write_json(output_path, results)
        print(
            f"  immediate {method:28s} "
            f"PPL={metrics['immediate']['perplexity']:.4f}",
            flush=True,
        )

    trajectories: List[Tuple[str, MaskDict | None]] = [(NO_COMPRESSION_METHOD, None)]
    trajectories.extend(masks.items())
    for method, method_masks in trajectories:
        if method_masks is None:
            model.load_state_dict(current_state, strict=True)
        else:
            restore_with_mask(
                model, current_state, reference_state, method_masks, args.device
            )
        optimizer = build_optimizer(model, current_optimizer_state)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.total_steps, eta_min=0.0
        )
        scheduler.load_state_dict(current_scheduler_state)
        continuation_metrics = train_segment(
            model,
            optimizer,
            scheduler,
            batches.continuation,
            seed + 1_000_000,
            args.device,
        )
        final_metrics = evaluate_lm(model, eval_batches, args.device)
        continuation_metrics["evaluation"] = final_metrics
        if method == NO_COMPRESSION_METHOD:
            seed_result["no_compression_continuation"] = continuation_metrics
            seed_result["no_compression_final"] = final_metrics
        else:
            seed_result["methods"][method]["continuation"] = continuation_metrics
            seed_result["methods"][method]["final"] = final_metrics
        del optimizer, scheduler
        model.zero_grad(set_to_none=True)
        empty_device_cache(args.device)
        write_json(output_path, results)
        print(f"  final     {method:28s} PPL={final_metrics['perplexity']:.4f}", flush=True)

    seed_result["wall_seconds"] = time.perf_counter() - seed_started
    del masks, components, components_raw, magnitude_scores, delta
    del current_state, current_optimizer_state, model
    gc.collect()
    empty_device_cache(args.device)
    write_json(output_path, results)


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    trust_radii = [
        float(value) for value in args.allocation_trust_radii.split(",") if value.strip()
    ]
    spectral_ranks = sorted(
        {int(value) for value in args.spectral_ranks.split(",") if value.strip()}
    )
    quantile_smoothness_values = sorted(
        {
            float(value)
            for value in args.quantile_smoothness_values.split(",")
            if value.strip()
        }
    )
    if not trust_radii or any(value <= 0 for value in trust_radii):
        raise ValueError("--allocation-trust-radii must contain positive values")
    if any(value > 1 for value in trust_radii):
        raise ValueError("--allocation-trust-radii values cannot exceed 1")
    if any(rank < 1 for rank in spectral_ranks):
        raise ValueError("--spectral-ranks must contain positive integers")
    if quantile_smoothness_values and any(
        not math.isfinite(value) or value <= 0 for value in quantile_smoothness_values
    ):
        raise ValueError("--quantile-smoothness-values must contain positive values")
    if quantile_smoothness_values and not 0 < args.quantile_trust_radius <= min(
        args.prune_ratio, args.max_layer_ratio - args.prune_ratio
    ):
        raise ValueError("--quantile-trust-radius must fit inside the layer-rate box")
    validate_args(args, seeds)
    configure_hf_offline()
    from dacp.models.gpt2 import get_gpt2_medium
    from dacp.utils.data_loader import _load_gpt2_tokenizer

    continuation_steps = args.total_steps - args.recovery_step
    started_at = datetime.now(timezone.utc)
    run_name = started_at.strftime("%Y%m%d_%H%M%S") + "_gpt2m_recovery_long"
    output_path = args.output_dir / f"{run_name}.json"
    results: Dict[str, object] = {
        "status": "started",
        "started_at": started_at.isoformat(),
        "config": {
            **vars(args),
            "checkpoint": str(args.checkpoint.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "seeds": seeds,
            "allocation_trust_radii": trust_radii,
            "spectral_ranks": spectral_ranks,
            "quantile_smoothness_values": quantile_smoothness_values,
            "scheduler": "cosine",
            "continuation_steps": continuation_steps,
        },
        "checkpoint": {
            "sha256": sha256_file(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "seed_results": {},
    }
    write_json(output_path, results)
    print(f"Writing incremental results to {output_path}", flush=True)

    checkpoint = load_training_checkpoint(args.checkpoint)
    reference_state = checkpoint.model_state
    reference_optimizer_state = checkpoint.optimizer_state
    del checkpoint
    tokenizer = _load_gpt2_tokenizer()
    training_pool = load_token_batches(
        args.data_dir / "train.txt",
        tokenizer,
        args.batch_size,
        args.seq_length,
        args.train_pool_batches,
    )
    eval_batches = load_token_batches(
        args.data_dir / "valid.txt",
        tokenizer,
        args.batch_size,
        args.seq_length,
        args.eval_batches,
    )
    results["evaluation_batches_sha256"] = batch_hash(eval_batches)
    context = LongRunContext(
        args=args,
        output_path=output_path,
        results=results,
        model_factory=lambda: get_gpt2_medium(pretrained=False),
        training_pool=training_pool,
        eval_batches=eval_batches,
        reference_state=reference_state,
        reference_optimizer_state=reference_optimizer_state,
        trust_radii=trust_radii,
        spectral_ranks=spectral_ranks,
        quantile_smoothness_values=quantile_smoothness_values,
    )
    wall_started = time.perf_counter()

    for seed_index, seed in enumerate(seeds, start=1):
        run_seed(context, seed, seed_index, len(seeds))

    results["aggregate"] = aggregate(results["seed_results"])
    results["status"] = "complete"
    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    results["wall_seconds"] = time.perf_counter() - wall_started
    write_json(output_path, results)
    print(f"\nComplete in {results['wall_seconds']:.1f}s: {output_path}", flush=True)


if __name__ == "__main__":
    main()
