"""Block-HVP Taylor scoring for residual checkpoint experiments."""

from __future__ import annotations

import time
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.lib.residual_masks import TensorDict
from experiments.lib.residual_runtime import empty_device_cache, lm_loss


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
            block_names = [
                name for name in named_params if name.startswith(layer_prefix)
            ]
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
                        signed.float().cpu(),
                        alpha=1.0 / len(batches),
                    )
                    if return_components:
                        first_order_accumulator[name].add_(
                            first_order.float().cpu(),
                            alpha=1.0 / len(batches),
                        )
                        second_order_accumulator[name].add_(
                            second_order.float().cpu(),
                            alpha=1.0 / len(batches),
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
    checksum_delta = [
        after - before for before, after in zip(checksum_before, checksum_after)
    ]
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


__all__ = [
    "compute_block_taylor_scores",
    "eligible_layers",
    "model_checksum",
]
