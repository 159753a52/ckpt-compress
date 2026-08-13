"""Execution state machine for the short residual allocation gate."""

from __future__ import annotations

import gc
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from experiments.lib.residual_masks import MaskDict, restore_with_mask
from experiments.lib.residual_protocol import NO_COMPRESSION_METHOD
from experiments.lib.residual_runtime import (
    batch_hash,
    checkpoint_optimizer_state,
    checkpoint_state,
    configure_hf_offline,
    empty_device_cache,
    evaluate_lm,
    lm_loss,
    load_token_batches,
    peak_memory_bytes,
    reset_peak_memory,
    set_seed,
    sha256_file,
    write_json,
)
from experiments.lib.residual_scoring import compute_block_taylor_scores
from experiments.lib.residual_short_config import parse_args, validate_short_config
from experiments.lib.residual_short_methods import (
    build_short_gate_masks,
    build_short_residual_scope,
    diagnose_short_gate_masks,
)


def continue_training(
    model: nn.Module,
    optimizer_state: Mapping,
    batches: Sequence[Mapping[str, torch.Tensor]],
    learning_rate: float,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    """Run an explicit, matched continuation trajectory after restoration."""
    set_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
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
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    configure_hf_offline()
    from dacp.models.gpt2 import get_gpt2_medium
    from dacp.utils.data_loader import _load_gpt2_tokenizer

    set_seed(args.seed)
    started_at = datetime.now(timezone.utc)
    run_name = started_at.strftime("%Y%m%d_%H%M%S") + f"_p{args.prune_ratio:.2f}"
    output_path = args.output_dir / f"{run_name}.json"
    results: Dict[str, Any] = {
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

    scope = build_short_residual_scope(
        model,
        current_state,
        reference_state,
        args.prune_ratio,
    )
    layers = scope.layers
    delta = scope.delta
    results["parameter_scope"] = scope.to_result_dict()
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
        model,
        train_batches[: args.hvp_batches],
        layers,
        delta,
        args.device,
    )
    scoring_metrics["peak_gpu_memory_bytes"] = peak_memory_bytes(args.device)
    results["scoring"] = scoring_metrics
    write_json(output_path, results)

    print("[5/6] Fitting layer Weibulls and constructing equal-budget masks", flush=True)
    masks, allocation = build_short_gate_masks(
        scope,
        taylor_scores,
        args.max_layer_ratio,
    )
    results["allocation"] = allocation
    results["methods"] = diagnose_short_gate_masks(scope, masks, taylor_scores)
    write_json(output_path, results)

    print("[6/6] Evaluating four restored snapshots", flush=True)
    for method, method_masks in masks.items():
        method_started = time.perf_counter()
        restore_with_mask(
            model,
            current_state,
            reference_state,
            method_masks,
            args.device,
        )
        metrics = evaluate_lm(model, eval_batches, args.device)
        results["methods"][method]["evaluation"] = metrics
        results["methods"][method]["restore_and_eval_seconds"] = (
            time.perf_counter() - method_started
        )
        write_json(output_path, results)
        print(
            f"  {method:28s} PPL={metrics['perplexity']:.4f} " f"loss={metrics['loss']:.6f}",
            flush=True,
        )

    if args.continuation_steps:
        print(
            f"[extra] Running matched {args.continuation_steps}-step continuation trajectories",
            flush=True,
        )
        optimizer_state = checkpoint_optimizer_state(args.current_checkpoint)
        trajectories: List[Tuple[str, MaskDict | None]] = [(NO_COMPRESSION_METHOD, None)]
        trajectories.extend(masks.items())
        for method, continuation_masks in trajectories:
            if continuation_masks is None:
                model.load_state_dict(current_state, strict=True)
            else:
                restore_with_mask(
                    model,
                    current_state,
                    reference_state,
                    continuation_masks,
                    args.device,
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
            if method == NO_COMPRESSION_METHOD:
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

    del masks, taylor_scores, scope, delta
    gc.collect()
    empty_device_cache(args.device)


__all__ = ["continue_training", "main", "parse_args", "validate_short_config"]


if __name__ == "__main__":
    main()
