"""Reusable residual checkpoint recovery and allocation diagnostics.

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

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.residual_allocation import (  # noqa: E402
    bounded_largest_remainder_counts,
    budget_tangent_dct_directions,
    calibrate_quantile_smooth_allocation,
    directional_layer_counts,
    fit_weibull_mom,
    largest_remainder_counts,
    reconstruct_directional_gradient,
    trust_region_counts,
    uniform_counts,
    weibull_cdf,
    weibull_counts,
)
from experiments.lib.residual_calibration import (  # noqa: E402
    batch_loss_values,
    calibrate_spectral_allocation,
    calibrate_trust_region_allocation,
)
from experiments.lib.residual_masks import (  # noqa: E402
    MaskDict,
    TensorDict,
    apply_layer_mask,
    apply_mask_from_device_states,
    cache_mask_states_on_device,
    exact_keep_mask,
    exact_keep_mask_from_order,
    global_mask,
    layer_mask_at_count,
    layer_masks,
    layer_rates,
    layer_score_orders,
    mask_metrics,
    mask_overlap,
    restore_with_mask,
)
from experiments.lib.residual_runtime import (  # noqa: E402
    LoadedTrainingCheckpoint,
    batch_hash,
    checkpoint_optimizer_state,
    checkpoint_state,
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
    synchronize_device,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    return parser.parse_args()


def continue_training(
    model: nn.Module,
    optimizer_state: Mapping,
    batches: Sequence[Mapping[str, torch.Tensor]],
    learning_rate: float,
    seed: int,
    device: str,
) -> Dict[str, object]:
    """Run an explicit, matched continuation trajectory after restoration."""
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer.load_state_dict(optimizer_state)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    losses = []
    started = time.perf_counter()
    model.train()
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = lm_loss(model, batch, device)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    seconds = time.perf_counter() - started
    del optimizer
    model.zero_grad(set_to_none=True)
    return {
        "steps": len(batches),
        "learning_rate": learning_rate,
        "seconds": seconds,
        "train_losses": losses,
    }


def eligible_layers(model: nn.Module) -> List[List[str]]:
    from dacp.tools.importance import build_transformer_blocks

    named_params = dict(model.named_parameters())
    layers = []
    for block in build_transformer_blocks(model, "gpt2"):
        eligible = [
            name
            for name in block
            if named_params[name].dim() >= 2
            and not any(
                pattern in name
                for pattern in ("wte", "wpe", "lm_head", "bias", "ln_", "LayerNorm")
            )
        ]
        if not eligible:
            raise RuntimeError(f"Transformer block has no eligible tensors: {block[:2]}")
        layers.append(eligible)
    return layers


def model_checksum(model: nn.Module) -> Tuple[float, float]:
    total = torch.zeros((), device=next(model.parameters()).device)
    absolute = torch.zeros_like(total)
    with torch.no_grad():
        for parameter in model.parameters():
            values = parameter.detach().float()
            total += values.sum()
            absolute += values.abs().sum()
    return total.item(), absolute.item()


def compute_block_taylor_scores(
    model: nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    layers: Sequence[Sequence[str]],
    delta: Mapping[str, torch.Tensor],
    device: str,
    return_components: bool = False,
) -> Tuple[TensorDict | Dict[str, TensorDict], Dict[str, object]]:
    """Compute g and H_bb*delta_b from one graph per Transformer layer."""
    named_params = dict(model.named_parameters())
    original_requires_grad = {name: p.requires_grad for name, p in named_params.items()}
    versions_before = {name: p._version for name, p in named_params.items()}
    checksum_before = model_checksum(model)
    scores: TensorDict = {}
    first_order_scores: TensorDict = {}
    second_order_scores: TensorDict = {}
    layer_seconds: List[float] = []
    started = time.perf_counter()

    model.eval()
    try:
        for layer_index, eligible_names in enumerate(layers):
            layer_prefix = f"transformer.h.{layer_index}."
            block_names = [name for name in named_params if name.startswith(layer_prefix)]
            block_set = set(block_names)
            for name, parameter in named_params.items():
                parameter.requires_grad_(name in block_set)

            block_params = [named_params[name] for name in block_names]
            block_started = time.perf_counter()
            signed_accumulator = {
                name: torch.zeros_like(delta[name], dtype=torch.float32)
                for name in eligible_names
            }
            if return_components:
                first_order_accumulator = {
                    name: torch.zeros_like(delta[name], dtype=torch.float32)
                    for name in eligible_names
                }
                second_order_accumulator = {
                    name: torch.zeros_like(delta[name], dtype=torch.float32)
                    for name in eligible_names
                }
            for batch in batches:
                model.zero_grad(set_to_none=True)
                with sdpa_kernel(SDPBackend.MATH):
                    loss = lm_loss(model, batch, device)
                    gradients = torch.autograd.grad(
                        loss,
                        block_params,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    gradient_by_name = dict(zip(block_names, gradients))
                    gradient_probe = loss.new_zeros(())
                    probes: Dict[str, torch.Tensor] = {}
                    for name in eligible_names:
                        gradient = gradient_by_name[name]
                        if gradient is None:
                            continue
                        probe = delta[name].to(device, non_blocking=True)
                        probes[name] = probe
                        gradient_probe = gradient_probe + (gradient * probe).sum()
                    hvps = torch.autograd.grad(
                        gradient_probe,
                        block_params,
                        retain_graph=False,
                        allow_unused=True,
                    )
                    hvp_by_name = dict(zip(block_names, hvps))

                for name in eligible_names:
                    gradient = gradient_by_name[name]
                    hvp = hvp_by_name[name]
                    if gradient is None or hvp is None:
                        continue
                    probe = probes[name]
                    first_order = -gradient.detach() * probe
                    second_order = 0.5 * probe * hvp.detach()
                    signed = first_order + second_order
                    signed_accumulator[name].add_(
                        signed.float().cpu(), alpha=1.0 / len(batches)
                    )
                    if return_components:
                        first_order_accumulator[name].add_(
                            first_order.float().cpu(), alpha=1.0 / len(batches)
                        )
                        second_order_accumulator[name].add_(
                            second_order.float().cpu(), alpha=1.0 / len(batches)
                        )
                del loss, gradients, gradient_by_name, gradient_probe, hvps, hvp_by_name, probes

            for name in eligible_names:
                scores[name] = signed_accumulator[name].abs()
                if return_components:
                    first_order_scores[name] = first_order_accumulator[name].abs()
                    second_order_scores[name] = second_order_accumulator[name].abs()

            layer_seconds.append(time.perf_counter() - block_started)
            print(
                f"  HVP layer {layer_index + 1:02d}/{len(layers):02d}: "
                f"{layer_seconds[-1]:.2f}s",
                flush=True,
            )
            del signed_accumulator
            if return_components:
                del first_order_accumulator, second_order_accumulator
            model.zero_grad(set_to_none=True)
            empty_device_cache(device)
    finally:
        for name, parameter in named_params.items():
            parameter.requires_grad_(original_requires_grad[name])

    versions_after = {name: p._version for name, p in named_params.items()}
    checksum_after = model_checksum(model)
    changed_versions = [
        name
        for name in versions_before
        if versions_before[name] != versions_after[name]
    ]
    checksum_delta = [after - before for before, after in zip(checksum_before, checksum_after)]
    if changed_versions or checksum_delta != [0.0, 0.0]:
        raise RuntimeError(
            f"Scoring mutated model weights: versions={changed_versions[:5]}, "
            f"checksum_delta={checksum_delta}"
        )

    output_scores: TensorDict | Dict[str, TensorDict]
    if return_components:
        output_scores = {
            "first_order": first_order_scores,
            "second_order": second_order_scores,
            "taylor": scores,
        }
    else:
        output_scores = scores

    return output_scores, {
        "total_seconds": time.perf_counter() - started,
        "layer_seconds": layer_seconds,
        "checksum_before": list(checksum_before),
        "checksum_after": list(checksum_after),
        "checksum_delta": checksum_delta,
        "changed_parameter_versions": changed_versions,
        "optimizer_constructed": False,
        "model_mode": "eval",
        "hvp_batches": len(batches),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.prune_ratio < 1:
        raise ValueError("--prune-ratio must be in (0, 1)")
    if args.hvp_batches < 1 or args.continuation_steps < 0 or args.train_batch_offset < 0:
        raise ValueError("HVP batches must be positive and step/offset counts non-negative")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")

    configure_hf_offline()
    from dacp.models.gpt2 import get_gpt2_medium
    from dacp.utils.data_loader import _load_gpt2_tokenizer

    set_seed(args.seed)
    started_at = datetime.now(timezone.utc)
    run_name = started_at.strftime("%Y%m%d_%H%M%S") + f"_p{args.prune_ratio:.2f}"
    output_path = args.output_dir / f"{run_name}.json"
    results: Dict[str, object] = {
        "status": "started",
        "started_at": started_at.isoformat(),
        "config": {
            "reference_checkpoint": str(args.reference_checkpoint.resolve()),
            "current_checkpoint": str(args.current_checkpoint.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "prune_ratio": args.prune_ratio,
            "max_layer_ratio": args.max_layer_ratio,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "eval_batches": args.eval_batches,
            "hvp_batches": args.hvp_batches,
            "train_batch_offset": args.train_batch_offset,
            "seed": args.seed,
            "device": args.device,
            "continuation_training_steps": args.continuation_steps,
            "continuation_learning_rate": args.continuation_lr,
        },
        "methods": {},
    }
    write_json(output_path, results)
    print(f"Writing incremental results to {output_path}", flush=True)

    wall_started = time.perf_counter()
    print("[1/6] Hashing and loading checkpoints", flush=True)
    results["checkpoints"] = {
        "reference_sha256": sha256_file(args.reference_checkpoint),
        "current_sha256": sha256_file(args.current_checkpoint),
        "reference_bytes": args.reference_checkpoint.stat().st_size,
        "current_bytes": args.current_checkpoint.stat().st_size,
    }
    reference_state = checkpoint_state(args.reference_checkpoint)
    current_state = checkpoint_state(args.current_checkpoint)

    print("[2/6] Loading current GPT-2 Medium snapshot and deterministic data", flush=True)
    model = get_gpt2_medium(pretrained=False)
    model.load_state_dict(current_state, strict=True)
    model.to(args.device).eval()
    tokenizer = _load_gpt2_tokenizer()
    train_batches = load_token_batches(
        args.data_dir / "train.txt",
        tokenizer,
        args.batch_size,
        args.seq_length,
        args.hvp_batches + args.continuation_steps,
        args.train_batch_offset,
    )
    eval_batches = load_token_batches(
        args.data_dir / "valid.txt",
        tokenizer,
        args.batch_size,
        args.seq_length,
        args.eval_batches,
    )
    results["data"] = {
        "hvp_batch_sha256": batch_hash(train_batches[: args.hvp_batches]),
        "continuation_batches_sha256": batch_hash(train_batches[args.hvp_batches :]),
        "eval_batches_sha256": batch_hash(eval_batches),
    }

    layers = eligible_layers(model)
    eligible_names = [name for layer in layers for name in layer]
    layer_sizes = [sum(current_state[name].numel() for name in layer) for layer in layers]
    eligible_count = sum(layer_sizes)
    model_count = sum(parameter.numel() for parameter in model.parameters())
    target_pruned = int(math.floor(args.prune_ratio * eligible_count))
    delta = {
        name: current_state[name].detach().float() - reference_state[name].detach().float()
        for name in eligible_names
    }
    magnitude_scores = {name: values.abs() for name, values in delta.items()}
    results["parameter_scope"] = {
        "transformer_layers": len(layers),
        "eligible_tensors": len(eligible_names),
        "eligible_parameters": eligible_count,
        "whole_model_parameters": model_count,
        "target_pruned": target_pruned,
        "target_eligible_sparsity": args.prune_ratio,
        "target_whole_model_sparsity": target_pruned / model_count,
        "layer_sizes": layer_sizes,
        "eligible_names": eligible_names,
    }
    write_json(output_path, results)

    print("[3/6] Evaluating pristine reference and current snapshots", flush=True)
    model.load_state_dict(reference_state, strict=True)
    reference_metrics = evaluate_lm(model, eval_batches, args.device)
    model.load_state_dict(current_state, strict=True)
    current_metrics = evaluate_lm(model, eval_batches, args.device)
    results["baselines"] = {"reference": reference_metrics, "current": current_metrics}
    write_json(output_path, results)
    print(
        f"  reference PPL={reference_metrics['perplexity']:.4f}; "
        f"current PPL={current_metrics['perplexity']:.4f}",
        flush=True,
    )

    print("[4/6] Computing one clean residual block-HVP", flush=True)
    reset_peak_memory(args.device)
    taylor_scores, scoring_metrics = compute_block_taylor_scores(
        model, train_batches[: args.hvp_batches], layers, delta, args.device
    )
    scoring_metrics["peak_gpu_memory_bytes"] = peak_memory_bytes(args.device)
    results["scoring"] = scoring_metrics
    write_json(output_path, results)

    print("[5/6] Fitting layer Weibulls and constructing equal-budget masks", flush=True)
    allocation_started = time.perf_counter()
    fits = []
    for index, layer in enumerate(layers):
        values = torch.cat([taylor_scores[name].flatten() for name in layer])
        fit = fit_weibull_mom(values)
        fit["layer"] = index
        fits.append(fit)
    uniform_layer_counts = uniform_counts(layer_sizes, target_pruned, args.prune_ratio)
    weibull_layer_counts, weibull_allocation = weibull_counts(
        fits,
        layer_sizes,
        target_pruned,
        args.prune_ratio,
        args.max_layer_ratio,
    )

    masks = {
        "residual_magnitude_uniform": layer_masks(
            layers, magnitude_scores, uniform_layer_counts
        ),
        "taylor_uniform": layer_masks(layers, taylor_scores, uniform_layer_counts),
        "taylor_weibull_mom": layer_masks(layers, taylor_scores, weibull_layer_counts),
        "taylor_exact_global": global_mask(eligible_names, taylor_scores, target_pruned),
    }
    results["allocation"] = {
        "seconds": time.perf_counter() - allocation_started,
        "weibull_fits": fits,
        "uniform_layer_counts": uniform_layer_counts,
        "weibull_layer_counts": weibull_layer_counts,
        "weibull": weibull_allocation,
    }

    exact_metrics = mask_metrics(masks["taylor_exact_global"], taylor_scores)
    for method, method_masks in masks.items():
        metrics = mask_metrics(method_masks, taylor_scores)
        metrics["eligible_sparsity"] = metrics["pruned"] / eligible_count
        metrics["whole_model_sparsity"] = metrics["pruned"] / model_count
        metrics["layer_rates"] = layer_rates(layers, method_masks)
        metrics["additive_regret_vs_exact"] = (
            (metrics["proxy_cost"] - exact_metrics["proxy_cost"])
            / max(abs(exact_metrics["proxy_cost"]), 1e-30)
        )
        metrics["overlap_with_exact"] = mask_overlap(
            method_masks, masks["taylor_exact_global"]
        )
        results["methods"][method] = metrics
    write_json(output_path, results)

    print("[6/6] Evaluating four restored snapshots", flush=True)
    for method, method_masks in masks.items():
        method_started = time.perf_counter()
        restore_with_mask(
            model, current_state, reference_state, method_masks, args.device
        )
        metrics = evaluate_lm(model, eval_batches, args.device)
        results["methods"][method]["evaluation"] = metrics
        results["methods"][method]["restore_and_eval_seconds"] = (
            time.perf_counter() - method_started
        )
        write_json(output_path, results)
        print(
            f"  {method:28s} PPL={metrics['perplexity']:.4f} "
            f"loss={metrics['loss']:.6f}",
            flush=True,
        )

    if args.continuation_steps:
        print(
            f"[extra] Running matched {args.continuation_steps}-step continuation trajectories",
            flush=True,
        )
        optimizer_state = checkpoint_optimizer_state(args.current_checkpoint)
        trajectories: List[Tuple[str, MaskDict | None]] = [("no_compression", None)]
        trajectories.extend(masks.items())
        for method, method_masks in trajectories:
            if method_masks is None:
                model.load_state_dict(current_state, strict=True)
            else:
                restore_with_mask(
                    model, current_state, reference_state, method_masks, args.device
                )
            continuation = continue_training(
                model,
                optimizer_state,
                train_batches[args.hvp_batches :],
                args.continuation_lr,
                args.seed + 1000,
                args.device,
            )
            continuation["evaluation"] = evaluate_lm(model, eval_batches, args.device)
            if method == "no_compression":
                results["continuation_no_compression"] = continuation
            else:
                results["methods"][method]["continuation"] = continuation
            write_json(output_path, results)
            print(
                f"  {method:28s} PPL={continuation['evaluation']['perplexity']:.4f}",
                flush=True,
            )
        del optimizer_state

    results["status"] = "complete"
    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    results["wall_seconds"] = time.perf_counter() - wall_started
    write_json(output_path, results)
    print(f"Complete in {results['wall_seconds']:.1f}s: {output_path}", flush=True)

    del masks, taylor_scores, magnitude_scores, delta
    gc.collect()
    empty_device_cache(args.device)


if __name__ == "__main__":
    main()
