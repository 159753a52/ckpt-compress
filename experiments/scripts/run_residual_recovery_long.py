"""Run a longer, paired residual checkpoint recovery experiment on one V100."""

from __future__ import annotations

import argparse
import copy
import gc
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.residual_allocation import (  # noqa: E402
    calibrate_quantile_smooth_allocation,
    fit_weibull_mom,
    uniform_counts,
    weibull_counts,
)
from experiments.lib.residual_calibration import (  # noqa: E402
    calibrate_spectral_allocation,
    calibrate_trust_region_allocation,
)
from experiments.lib.residual_masks import (  # noqa: E402
    MaskDict,
    TensorDict,
    global_mask,
    layer_masks,
    layer_rates,
    layer_score_orders,
    mask_metrics,
    mask_overlap,
    restore_with_mask,
)
from experiments.lib.residual_recovery import (  # noqa: E402
    compute_block_taylor_scores,
    eligible_layers,
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
    set_seed,
    sha256_file,
    write_json,
)


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


def clone_model_state_to_cpu(model: torch.nn.Module) -> TensorDict:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def seeded_training_batches(
    pool: Sequence[Mapping[str, torch.Tensor]],
    count: int,
    seed: int,
) -> Tuple[List[Mapping[str, torch.Tensor]], List[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(pool), generator=generator)[:count].tolist()
    return [pool[index] for index in indices], indices


def build_optimizer(
    model: torch.nn.Module,
    state: Mapping,
    learning_rate: float | None = None,
) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    optimizer.load_state_dict(state)
    if learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
            group["initial_lr"] = learning_rate
    return optimizer


def train_segment(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batches: Sequence[Mapping[str, torch.Tensor]],
    seed: int,
    device: str,
) -> Dict[str, object]:
    set_seed(seed)
    model.train()
    losses = []
    learning_rates = []
    started = time.perf_counter()
    for batch in batches:
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.zero_grad(set_to_none=True)
        loss = lm_loss(model, batch, device)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return {
        "steps": len(batches),
        "seconds": time.perf_counter() - started,
        "train_losses": losses,
        "learning_rates": learning_rates,
    }


def build_masks(
    layers: Sequence[Sequence[str]],
    magnitude_scores: TensorDict,
    components: Dict[str, TensorDict],
    prune_ratio: float,
    max_layer_ratio: float,
    taylor_score_orders: Sequence[torch.Tensor] | None = None,
) -> Tuple[Dict[str, MaskDict], Dict[str, object]]:
    taylor_scores = components["taylor"]
    layer_sizes = [sum(taylor_scores[name].numel() for name in layer) for layer in layers]
    eligible_count = sum(layer_sizes)
    target = int(math.floor(prune_ratio * eligible_count))
    uniform_layer_counts = uniform_counts(layer_sizes, target, prune_ratio)

    fits = []
    for index, layer in enumerate(layers):
        values = torch.cat([taylor_scores[name].flatten() for name in layer])
        fit = fit_weibull_mom(values)
        fit["layer"] = index
        fits.append(fit)
    weibull_layer_counts, weibull_metadata = weibull_counts(
        fits, layer_sizes, target, prune_ratio, max_layer_ratio
    )
    eligible_names = [name for layer in layers for name in layer]
    masks = {
        "residual_magnitude_uniform": layer_masks(
            layers, magnitude_scores, uniform_layer_counts
        ),
        "first_order_uniform": layer_masks(
            layers, components["first_order"], uniform_layer_counts
        ),
        "second_order_uniform": layer_masks(
            layers, components["second_order"], uniform_layer_counts
        ),
        "taylor_uniform": layer_masks(
            layers, taylor_scores, uniform_layer_counts, taylor_score_orders
        ),
        "taylor_weibull_mom": layer_masks(
            layers, taylor_scores, weibull_layer_counts, taylor_score_orders
        ),
        "taylor_exact_global": global_mask(eligible_names, taylor_scores, target),
    }
    metadata = {
        "eligible_parameters": eligible_count,
        "target_pruned": target,
        "target_eligible_sparsity": prune_ratio,
        "layer_sizes": layer_sizes,
        "uniform_layer_counts": uniform_layer_counts,
        "weibull_layer_counts": weibull_layer_counts,
        "weibull_fits": fits,
        "weibull": weibull_metadata,
    }
    return masks, metadata


def score_for_method(
    method: str,
    magnitude_scores: TensorDict,
    components: Dict[str, TensorDict],
) -> TensorDict:
    if method == "residual_magnitude_uniform":
        return magnitude_scores
    if method == "first_order_uniform":
        return components["first_order"]
    if method == "second_order_uniform":
        return components["second_order"]
    return components["taylor"]


def summarize_values(values: Sequence[float]) -> Dict[str, object]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return {"values": list(values), "mean": mean, "std": None, "ci95": None}
    std = statistics.stdev(values)
    half_width = float(student_t.ppf(0.975, len(values) - 1)) * std / math.sqrt(len(values))
    return {
        "values": list(values),
        "mean": mean,
        "std": std,
        "ci95": [mean - half_width, mean + half_width],
    }


def float_slug(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p").replace("+", "")


def aggregate(seed_results: Mapping[str, Mapping]) -> Dict[str, object]:
    seeds = sorted(seed_results, key=int)
    method_names = list(seed_results[seeds[0]]["methods"])
    output: Dict[str, object] = {
        "current_perplexity": summarize_values(
            [seed_results[seed]["current"]["perplexity"] for seed in seeds]
        ),
        "no_compression_final_perplexity": summarize_values(
            [seed_results[seed]["no_compression_final"]["perplexity"] for seed in seeds]
        ),
        "methods": {},
    }
    for method in method_names:
        immediate = [
            seed_results[seed]["methods"][method]["immediate"]["perplexity"]
            for seed in seeds
        ]
        final = [
            seed_results[seed]["methods"][method]["final"]["perplexity"]
            for seed in seeds
        ]
        magnitude_immediate = [
            seed_results[seed]["methods"]["residual_magnitude_uniform"]["immediate"]["perplexity"]
            for seed in seeds
        ]
        magnitude_final = [
            seed_results[seed]["methods"]["residual_magnitude_uniform"]["final"]["perplexity"]
            for seed in seeds
        ]
        output["methods"][method] = {
            "immediate_perplexity": summarize_values(immediate),
            "final_perplexity": summarize_values(final),
            "paired_immediate_delta_vs_magnitude": summarize_values(
                [value - baseline for value, baseline in zip(immediate, magnitude_immediate)]
            ),
            "paired_final_delta_vs_magnitude": summarize_values(
                [value - baseline for value, baseline in zip(final, magnitude_final)]
            ),
        }
    output["paired_comparisons"] = {}
    comparisons = {
        "taylor_vs_first_order": ("taylor_uniform", "first_order_uniform"),
        "second_order_vs_first_order": ("second_order_uniform", "first_order_uniform"),
        "weibull_vs_taylor_uniform": ("taylor_weibull_mom", "taylor_uniform"),
        "exact_global_vs_taylor_uniform": ("taylor_exact_global", "taylor_uniform"),
        "probe_trust_vs_taylor_uniform": ("taylor_probe_trust", "taylor_uniform"),
        "probe_trust_vs_weibull": ("taylor_probe_trust", "taylor_weibull_mom"),
    }
    for method in method_names:
        if not method.startswith(("taylor_spectral_k", "taylor_quantile_")):
            continue
        label = method.removeprefix("taylor_")
        comparisons[f"{label}_vs_taylor_uniform"] = (method, "taylor_uniform")
        comparisons[f"{label}_vs_weibull"] = (method, "taylor_weibull_mom")
        comparisons[f"{label}_vs_probe_trust"] = (method, "taylor_probe_trust")
    for label, (left, right) in comparisons.items():
        comparison = {}
        for stage in ("immediate", "final"):
            deltas = [
                seed_results[seed]["methods"][left][stage]["perplexity"]
                - seed_results[seed]["methods"][right][stage]["perplexity"]
                for seed in seeds
            ]
            comparison[f"{stage}_perplexity_delta"] = summarize_values(deltas)
        output["paired_comparisons"][label] = comparison
    return output


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
    selected_batch_count = (
        args.total_steps
        + args.hvp_batches
        + args.allocation_probe_batches
        + args.allocation_selection_batches
    )
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
    wall_started = time.perf_counter()

    for seed_index, seed in enumerate(seeds, start=1):
        print(f"\n=== Seed {seed} ({seed_index}/{len(seeds)}) ===", flush=True)
        seed_started = time.perf_counter()
        selected, selected_indices = seeded_training_batches(
            training_pool, selected_batch_count, seed
        )
        pre_batches = selected[: args.recovery_step]
        score_start = args.recovery_step
        probe_start = score_start + args.hvp_batches
        selection_start = probe_start + args.allocation_probe_batches
        continuation_start = selection_start + args.allocation_selection_batches
        score_batches = selected[score_start:probe_start]
        probe_batches = selected[probe_start:selection_start]
        selection_batches = selected[selection_start:continuation_start]
        continuation_batches = selected[continuation_start:]
        if len(continuation_batches) != continuation_steps:
            raise RuntimeError("Internal batch partition does not match continuation steps")

        seed_result: Dict[str, object] = {
            "seed": seed,
            "selected_pool_indices": selected_indices,
            "data_hashes": {
                "pre_recovery": batch_hash(pre_batches),
                "scoring": batch_hash(score_batches),
                "allocation_probe": batch_hash(probe_batches),
                "allocation_selection": batch_hash(selection_batches),
                "continuation": batch_hash(continuation_batches),
            },
            "methods": {},
        }
        results["seed_results"][str(seed)] = seed_result
        write_json(output_path, results)

        model = get_gpt2_medium(pretrained=False)
        model.load_state_dict(reference_state, strict=True)
        model.to(args.device)
        reference_metrics = evaluate_lm(model, eval_batches, args.device)
        seed_result["reference"] = reference_metrics

        optimizer = build_optimizer(
            model, reference_optimizer_state, learning_rate=args.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.total_steps, eta_min=0.0
        )
        pre_metrics = train_segment(
            model, optimizer, scheduler, pre_batches, seed, args.device
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
            score_batches,
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
        if spectral_ranks or quantile_smoothness_values:
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
            probe_batches,
            selection_batches,
            args.allocation_probe_radius,
            trust_radii,
            args.device,
            taylor_score_orders,
        )
        masks["taylor_probe_trust"] = layer_masks(
            layers, components["taylor"], probe_trust_counts, taylor_score_orders
        )
        allocation_metadata["probe_trust_layer_counts"] = probe_trust_counts
        allocation_metadata["probe_trust"] = probe_trust_metadata
        if spectral_ranks:
            spectral_counts, spectral_metadata = calibrate_spectral_allocation(
                model,
                current_state,
                reference_state,
                layers,
                components["taylor"],
                allocation_metadata["target_pruned"],
                args.prune_ratio,
                args.max_layer_ratio,
                probe_batches,
                args.spectral_probe_radius,
                spectral_ranks,
                args.spectral_trust_radius,
                args.device,
                taylor_score_orders,
            )
            for rank, counts in spectral_counts.items():
                masks[f"taylor_spectral_k{rank}"] = layer_masks(
                    layers, components["taylor"], counts, taylor_score_orders
                )
            allocation_metadata["spectral_layer_counts"] = spectral_counts
            allocation_metadata["spectral"] = spectral_metadata
        if quantile_smoothness_values:
            quantile_counts, quantile_metadata = calibrate_quantile_smooth_allocation(
                layers,
                components["taylor"],
                taylor_score_orders,
                allocation_metadata["target_pruned"],
                args.prune_ratio,
                args.max_layer_ratio,
                args.quantile_trust_radius,
                quantile_smoothness_values,
                args.device,
                args.quantile_cost_normalization,
            )
            for smoothness, counts in quantile_counts.items():
                normalization_slug = (
                    "relative"
                    if args.quantile_cost_normalization == "layer_uniform_cost"
                    else "global"
                )
                method = (
                    f"taylor_quantile_{normalization_slug}_smooth_"
                    f"l{float_slug(smoothness)}"
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
        exact_metrics = mask_metrics(masks["taylor_exact_global"], components["taylor"])

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
                method_masks, masks["taylor_exact_global"]
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

        trajectories: List[Tuple[str, MaskDict | None]] = [("no_compression", None)]
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
                continuation_batches,
                seed + 1_000_000,
                args.device,
            )
            final_metrics = evaluate_lm(model, eval_batches, args.device)
            continuation_metrics["evaluation"] = final_metrics
            if method == "no_compression":
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

    results["aggregate"] = aggregate(results["seed_results"])
    results["status"] = "complete"
    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    results["wall_seconds"] = time.perf_counter() - wall_started
    write_json(output_path, results)
    print(f"\nComplete in {results['wall_seconds']:.1f}s: {output_path}", flush=True)


if __name__ == "__main__":
    main()
