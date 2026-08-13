"""Shared quantize-dequantize adapters used by experiment runners."""

from __future__ import annotations

from typing import Mapping, TypeAlias

import torch

from dacp.quantization import INT4Quantizer, KMeansQuantizer

ExperimentQuantizer: TypeAlias = KMeansQuantizer | INT4Quantizer
DEFAULT_SKIP_PATTERNS = ("embed", "wte", "wpe", "ln_", "LayerNorm", "layernorm")


def quantize_dequantize_tensor(
    weight: torch.Tensor,
    quantizer: ExperimentQuantizer,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Round-trip one tensor while keeping pruned positions exactly zero."""
    if mask is not None:
        if mask.shape != weight.shape:
            raise ValueError("quantization mask shape must match the weight")
        if mask.device != weight.device:
            raise ValueError("quantization mask and weight must be on the same device")
        if mask.dtype != torch.bool and not torch.all((mask == 0) | (mask == 1)).item():
            raise ValueError("quantization mask must contain only 0 or 1")

    if isinstance(quantizer, KMeansQuantizer):
        quantized, metadata = quantizer.quantize(weight, mask=mask)
    elif isinstance(quantizer, INT4Quantizer):
        quantized, metadata = quantizer.quantize(weight)
    else:
        raise TypeError(f"Unsupported experiment quantizer: {type(quantizer).__name__}")

    recovered = quantizer.dequantize(quantized, metadata).to(weight.device)
    if mask is not None:
        recovered = recovered * mask.to(dtype=recovered.dtype)
    return recovered


def quantize_model_parameters(
    model: torch.nn.Module,
    masks: Mapping[str, torch.Tensor],
    quantizer: ExperimentQuantizer,
    *,
    skip_patterns: tuple[str, ...] = DEFAULT_SKIP_PATTERNS,
) -> None:
    """Round-trip matrix parameters and preserve all supplied pruning masks."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.dim() < 2 or any(pattern in name for pattern in skip_patterns):
                continue
            recovered = quantize_dequantize_tensor(
                parameter,
                quantizer,
                mask=masks.get(name),
            )
            parameter.copy_(recovered)


__all__ = [
    "DEFAULT_SKIP_PATTERNS",
    "ExperimentQuantizer",
    "quantize_dequantize_tensor",
    "quantize_model_parameters",
]
