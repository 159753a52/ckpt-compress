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

import gc
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


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
from experiments.lib.residual_training import (  # noqa: E402
    SeedBatchPartition,
    build_optimizer,
    clone_model_state_to_cpu,
    partition_seed_batches,
    seeded_training_batches,
    train_segment,
)
from experiments.lib.residual_scoring import (  # noqa: E402
    compute_block_taylor_scores,
    eligible_layers,
    model_checksum,
)
from experiments.lib.residual_short_config import (  # noqa: E402
    parse_args,
    validate_short_config,
)


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


def main() -> None:
    args = parse_args()
    validate_short_config(args)
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
