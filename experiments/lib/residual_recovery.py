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
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import brentq
from scipy.special import gammaln
from torch.nn.attention import SDPBackend, sdpa_kernel


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dacp.models.gpt2 import get_gpt2_medium  # noqa: E402
from dacp.tools.importance import build_transformer_blocks  # noqa: E402
from dacp.utils.data_loader import _load_gpt2_tokenizer  # noqa: E402


TensorDict = Dict[str, torch.Tensor]
MaskDict = Dict[str, torch.Tensor]


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_state(path: Path) -> TensorDict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload in {path}: {type(payload)}")
    for key in ("model_state_dict", "state_dict", "model"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise KeyError(f"No model state dict found in {path}")


def checkpoint_optimizer_state(path: Path) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "optimizer_state_dict" not in payload:
        raise KeyError(f"No optimizer state dict found in {path}")
    return payload["optimizer_state_dict"]


def optimizer_state_to_cpu(state: Mapping) -> Dict:
    """Clone an optimizer state dict to CPU without a transient GPU deepcopy."""
    result = {"state": {}, "param_groups": copy.deepcopy(state["param_groups"])}
    for parameter_id, parameter_state in state["state"].items():
        result["state"][parameter_id] = {
            key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in parameter_state.items()
        }
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def load_token_batches(
    path: Path,
    tokenizer,
    batch_size: int,
    seq_length: int,
    num_batches: int,
    batch_offset: int = 0,
) -> List[Dict[str, torch.Tensor]]:
    skip = batch_size * seq_length * batch_offset
    needed = batch_size * seq_length * num_batches
    required = skip + needed
    tokens: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tokens.extend(tokenizer.encode(line))
            if len(tokens) >= required:
                break
    if len(tokens) < required:
        raise RuntimeError(f"Only found {len(tokens)} tokens in {path}, need {required}")

    batches = []
    offset = skip
    for _ in range(num_batches):
        size = batch_size * seq_length
        input_ids = torch.tensor(tokens[offset : offset + size], dtype=torch.long)
        input_ids = input_ids.view(batch_size, seq_length)
        batches.append({"input_ids": input_ids, "labels": input_ids.clone()})
        offset += size
    return batches


def batch_hash(batches: Sequence[Mapping[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(batch["input_ids"].contiguous().numpy().tobytes())
    return digest.hexdigest()


def lm_loss(model: nn.Module, batch: Mapping[str, torch.Tensor], device: str) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    outputs = model(input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    return nn.functional.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
        labels[..., 1:].contiguous().view(-1),
    )


def evaluate_lm(
    model: nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    device: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    started = time.perf_counter()
    with torch.no_grad():
        for batch in batches:
            total_loss += lm_loss(model, batch, device).item()
    average = total_loss / len(batches)
    return {
        "loss": average,
        "perplexity": math.exp(min(average, 20.0)),
        "seconds": time.perf_counter() - started,
        "batches": len(batches),
        "tokens": sum(batch["input_ids"].numel() for batch in batches),
    }


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
            torch.cuda.empty_cache()
    finally:
        for name, parameter in named_params.items():
            parameter.requires_grad_(original_requires_grad[name])

    versions_after = {name: p._version for name, p in named_params.items()}
    checksum_after = model_checksum(model)
    changed_versions = [name for name in versions_before if versions_before[name] != versions_after[name]]
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


def exact_keep_mask(values: torch.Tensor, prune_count: int) -> torch.Tensor:
    """Return a deterministic mask with exactly prune_count False entries."""
    flat = values.detach().float().flatten().cpu()
    count = flat.numel()
    if prune_count < 0 or prune_count > count:
        raise ValueError(f"Invalid prune count {prune_count} for {count} scores")
    if prune_count == 0:
        return torch.ones(count, dtype=torch.bool)
    if prune_count == count:
        return torch.zeros(count, dtype=torch.bool)

    threshold = torch.kthvalue(flat, prune_count).values
    pruned = flat < threshold
    remaining = prune_count - int(pruned.sum().item())
    if remaining:
        tied = torch.nonzero(flat == threshold, as_tuple=False).flatten()
        pruned[tied[:remaining]] = True
    if int(pruned.sum().item()) != prune_count:
        raise RuntimeError("Tie handling failed to produce the requested prune count")
    return ~pruned


def largest_remainder_counts(
    real_counts: Sequence[float],
    target: int,
    capacities: Sequence[int],
) -> List[int]:
    counts = [min(int(math.floor(value)), cap) for value, cap in zip(real_counts, capacities)]
    difference = target - sum(counts)
    if difference > 0:
        order = sorted(
            range(len(counts)),
            key=lambda i: (real_counts[i] - math.floor(real_counts[i]), -i),
            reverse=True,
        )
        while difference:
            progressed = False
            for index in order:
                if counts[index] < capacities[index]:
                    counts[index] += 1
                    difference -= 1
                    progressed = True
                    if difference == 0:
                        break
            if not progressed:
                raise RuntimeError("Layer capacities cannot meet the global pruning target")
    elif difference < 0:
        order = sorted(
            range(len(counts)),
            key=lambda i: (real_counts[i] - math.floor(real_counts[i]), -i),
        )
        while difference:
            progressed = False
            for index in order:
                if counts[index] > 0:
                    counts[index] -= 1
                    difference += 1
                    progressed = True
                    if difference == 0:
                        break
            if not progressed:
                raise RuntimeError("Could not round layer counts to the global target")
    return counts


def uniform_counts(layer_sizes: Sequence[int], target: int, ratio: float) -> List[int]:
    real = [ratio * size for size in layer_sizes]
    return largest_remainder_counts(real, target, list(layer_sizes))


def fit_weibull_mom(values: torch.Tensor) -> Dict[str, float | bool | str]:
    flat = values.detach().float().flatten().cpu()
    count = flat.numel()
    maximum = flat.max().item()
    zero_fraction = (flat == 0).sum().item() / count
    if not math.isfinite(maximum) or maximum <= 0:
        return {"valid": False, "reason": "non-positive maximum", "zero_fraction": zero_fraction}

    scaled = flat / maximum
    total = scaled.sum(dtype=torch.float64).item()
    total_squared = (scaled * scaled).sum(dtype=torch.float64).item()
    mean = total / count
    variance = max(total_squared / count - mean * mean, 0.0)
    if mean <= 1e-15 or variance <= 0:
        return {"valid": False, "reason": "degenerate moments", "zero_fraction": zero_fraction}
    cv_squared = variance / (mean * mean)
    if cv_squared < 1e-10 or not math.isfinite(cv_squared):
        return {"valid": False, "reason": "degenerate coefficient of variation", "zero_fraction": zero_fraction}

    log_target = math.log1p(cv_squared)

    def objective(shape: float) -> float:
        return float(gammaln(1.0 + 2.0 / shape) - 2.0 * gammaln(1.0 + 1.0 / shape) - log_target)

    try:
        shape = brentq(objective, 0.01, 1000.0, maxiter=200)
    except ValueError:
        return {"valid": False, "reason": "shape root not bracketed", "zero_fraction": zero_fraction}
    log_scale = math.log(mean) - float(gammaln(1.0 + 1.0 / shape)) + math.log(maximum)
    scale = math.exp(log_scale)
    if not math.isfinite(scale) or scale <= 0:
        return {"valid": False, "reason": "invalid scale", "zero_fraction": zero_fraction}
    return {
        "valid": True,
        "count": count,
        "mean": mean * maximum,
        "variance": variance * maximum * maximum,
        "cv_squared": cv_squared,
        "shape": shape,
        "scale": scale,
        "zero_fraction": zero_fraction,
    }


def weibull_cdf(threshold: float, fit: Mapping[str, float | bool | str]) -> float:
    if threshold <= 0:
        return 0.0
    shape = float(fit["shape"])
    scale = float(fit["scale"])
    log_power = shape * (math.log(threshold) - math.log(scale))
    if log_power > 40:
        return 1.0
    if log_power < -40:
        return math.exp(log_power)
    return -math.expm1(-math.exp(log_power))


def weibull_counts(
    fits: Sequence[Mapping[str, float | bool | str]],
    layer_sizes: Sequence[int],
    target: int,
    ratio: float,
    max_layer_ratio: float,
) -> Tuple[List[int], Dict[str, object]]:
    capacities = [min(size, int(math.floor(max_layer_ratio * size))) for size in layer_sizes]
    if sum(capacities) < target:
        raise ValueError("max_layer_ratio makes the requested global ratio infeasible")

    def real_counts_at(threshold: float) -> List[float]:
        result = []
        for fit, size, capacity in zip(fits, layer_sizes, capacities):
            if bool(fit["valid"]):
                count = size * weibull_cdf(threshold, fit)
            else:
                count = size * ratio
            result.append(min(count, float(capacity)))
        return result

    valid_scales = [float(fit["scale"]) for fit in fits if bool(fit["valid"])]
    if not valid_scales:
        counts = uniform_counts(layer_sizes, target, ratio)
        return counts, {"threshold": None, "fallback": "all Weibull fits invalid"}

    high = max(valid_scales)
    while sum(real_counts_at(high)) < target:
        high *= 2.0
    threshold = brentq(
        lambda value: sum(real_counts_at(value)) - target,
        0.0,
        high,
        maxiter=200,
    )
    real_counts = real_counts_at(threshold)
    counts = largest_remainder_counts(real_counts, target, capacities)
    return counts, {
        "threshold": threshold,
        "real_counts": real_counts,
        "capacities": capacities,
        "fallback": None,
    }


def layer_masks(
    layers: Sequence[Sequence[str]],
    scores: Mapping[str, torch.Tensor],
    counts: Sequence[int],
) -> MaskDict:
    masks: MaskDict = {}
    for names, prune_count in zip(layers, counts):
        flat_scores = torch.cat([scores[name].flatten() for name in names])
        flat_keep = exact_keep_mask(flat_scores, prune_count)
        offset = 0
        for name in names:
            size = scores[name].numel()
            masks[name] = flat_keep[offset : offset + size].view_as(scores[name])
            offset += size
    return masks


def global_mask(
    names: Sequence[str],
    scores: Mapping[str, torch.Tensor],
    prune_count: int,
) -> MaskDict:
    all_scores = torch.cat([scores[name].flatten() for name in names])
    all_keep = exact_keep_mask(all_scores, prune_count)
    masks: MaskDict = {}
    offset = 0
    for name in names:
        size = scores[name].numel()
        masks[name] = all_keep[offset : offset + size].view_as(scores[name])
        offset += size
    return masks


def mask_metrics(
    masks: Mapping[str, torch.Tensor],
    taylor_scores: Mapping[str, torch.Tensor],
) -> Dict[str, float | int]:
    pruned = 0
    proxy_cost = 0.0
    for name, keep in masks.items():
        removed = ~keep
        pruned += removed.sum().item()
        proxy_cost += taylor_scores[name][removed].sum(dtype=torch.float64).item()
    return {"pruned": int(pruned), "proxy_cost": proxy_cost}


def mask_overlap(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> Dict[str, float | int]:
    intersection = 0
    union = 0
    disagreements = 0
    for name in left:
        left_pruned = ~left[name]
        right_pruned = ~right[name]
        intersection += (left_pruned & right_pruned).sum().item()
        union += (left_pruned | right_pruned).sum().item()
        disagreements += (left_pruned != right_pruned).sum().item()
    return {
        "pruned_jaccard": intersection / max(union, 1),
        "mask_disagreements": int(disagreements),
    }


def restore_with_mask(
    model: nn.Module,
    current_state: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    device: str,
) -> None:
    model.load_state_dict(current_state, strict=True)
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, keep in masks.items():
            reference = reference_state[name].to(device, non_blocking=True)
            current = current_state[name].to(device, non_blocking=True)
            restored = torch.where(keep.to(device, non_blocking=True), current, reference)
            named_params[name].copy_(restored)


def layer_rates(layers: Sequence[Sequence[str]], masks: Mapping[str, torch.Tensor]) -> List[float]:
    rates = []
    for names in layers:
        total = sum(masks[name].numel() for name in names)
        pruned = sum((~masks[name]).sum().item() for name in names)
        rates.append(pruned / total)
    return rates


def main() -> None:
    args = parse_args()
    if not 0 < args.prune_ratio < 1:
        raise ValueError("--prune-ratio must be in (0, 1)")
    if args.hvp_batches < 1 or args.continuation_steps < 0 or args.train_batch_offset < 0:
        raise ValueError("HVP batches must be positive and step/offset counts non-negative")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")

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
    torch.cuda.reset_peak_memory_stats()
    taylor_scores, scoring_metrics = compute_block_taylor_scores(
        model, train_batches[: args.hvp_batches], layers, delta, args.device
    )
    scoring_metrics["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
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
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
