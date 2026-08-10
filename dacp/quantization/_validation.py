"""Shared validation and error metrics for quantizers."""

from __future__ import annotations

import torch


def validate_quantization_input(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Require a non-empty finite floating-point tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must be a floating-point tensor")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def relative_mse(original: torch.Tensor, recovered: torch.Tensor) -> float:
    """Return MSE normalized by population variance with stable scalar behavior."""
    validate_quantization_input("original", original)
    validate_quantization_input("recovered", recovered)
    if original.shape != recovered.shape:
        raise ValueError(
            "original and recovered shapes must match: "
            f"{tuple(original.shape)} != {tuple(recovered.shape)}"
        )

    original_float = original.detach().float()
    recovered_float = recovered.detach().to(original_float.device).float()
    mse = torch.mean((original_float - recovered_float) ** 2)
    centered = original_float - original_float.mean()
    variance = torch.mean(centered ** 2)
    return (mse / (variance + 1e-8)).item()


__all__ = ["relative_mse", "validate_quantization_input"]
