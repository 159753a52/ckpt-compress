"""Measure exact serialized sizes for a top-k residual checkpoint stream.

This is a storage diagnostic, not a quality experiment.  Both checkpoints must
contain exactly the same finite floating-point tensors so the denominator and
the encoded residual cover the same parameter scope.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dacp.pruning.masks import exact_keep_mask


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    """Load a tensor-only state dict from a trusted project checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint {path} must contain a mapping")
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            checkpoint = candidate
            break
    if not checkpoint:
        raise ValueError(f"Checkpoint {path} contains an empty state dict")
    if not all(
        isinstance(name, str) and torch.is_tensor(value) for name, value in checkpoint.items()
    ):
        raise TypeError(f"State dict in {path} must map string names to tensors")
    state = {
        name: value.detach().cpu()
        for name, value in checkpoint.items()
        if value.is_floating_point()
    }
    if not state:
        raise ValueError(f"Checkpoint {path} contains no floating-point tensors")
    for name, value in state.items():
        if value.is_complex() or not torch.isfinite(value).all().item():
            raise ValueError(f"Tensor {name!r} in {path} must be finite real floating point")
    return state


def residuals(
    newer: Mapping[str, torch.Tensor],
    older: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Validate checkpoint compatibility and return flattened fp32 residuals."""
    missing = sorted(set(newer).difference(older))
    extra = sorted(set(older).difference(newer))
    if missing or extra:
        raise ValueError(f"Checkpoint tensor keys must match: missing={missing}, extra={extra}")
    result: Dict[str, torch.Tensor] = {}
    for name, current in newer.items():
        reference = older[name]
        if current.shape != reference.shape:
            raise ValueError(
                f"Checkpoint tensor shape differs for {name!r}: "
                f"{tuple(current.shape)} != {tuple(reference.shape)}"
            )
        result[name] = (current.float() - reference.float()).flatten()
    return result


def parse_ratios(raw: str) -> list[float]:
    """Parse a non-empty, unique list of finite prune ratios."""
    try:
        ratios = [float(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"ratios must be comma-separated numbers, got {raw!r}") from exc
    if not ratios:
        raise ValueError("ratios must contain at least one value")
    if any(not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0 for ratio in ratios):
        raise ValueError("ratios must contain only finite values in [0, 1]")
    if len(set(ratios)) != len(ratios):
        raise ValueError("ratios must not contain duplicates")
    return ratios


def exact_residual_masks(
    deltas: Mapping[str, torch.Tensor],
    prune_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Construct deterministic per-tensor keep masks for one exact global budget."""
    if not deltas:
        raise ValueError("deltas must contain at least one tensor")
    if not math.isfinite(prune_ratio) or not 0.0 <= prune_ratio <= 1.0:
        raise ValueError(f"prune_ratio must be finite and in [0, 1], got {prune_ratio}")
    names = list(deltas)
    sizes = [deltas[name].numel() for name in names]
    if any(size == 0 for size in sizes):
        raise ValueError("Residual tensors must be non-empty")
    scores = torch.cat([deltas[name].abs().float() for name in names])
    prune_count = math.floor(scores.numel() * prune_ratio)
    global_keep = exact_keep_mask(scores, prune_count).flatten()
    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, size in zip(names, sizes):
        masks[name] = global_keep[offset : offset + size]
        offset += size
    if sum(int((~mask).sum().item()) for mask in masks.values()) != prune_count:
        raise RuntimeError("Global residual mask did not preserve the exact pruning budget")
    return masks


def int4_pack(values: torch.Tensor) -> tuple[bytes, float]:
    """Symmetric per-tensor INT4 quantization, packed two values per byte."""
    if values.numel() == 0:
        return b"", 1.0
    scale = values.abs().max().item() / 7.0 or 1.0
    quantized = torch.clamp(torch.round(values / scale), -8, 7).to(torch.int8) + 8
    array = quantized.numpy().astype(np.uint8)
    if len(array) % 2:
        array = np.append(array, 0)
    packed = (array[0::2] << 4) | array[1::2]
    return packed.tobytes(), scale


def serialize_residual(
    deltas: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    dtype: str,
) -> bytes:
    """Serialize masks, surviving values, and one scale per tensor."""
    if dtype not in {"int4", "fp16"}:
        raise ValueError(f"Unsupported residual dtype: {dtype!r}")
    if set(deltas) != set(masks):
        raise ValueError("Residual deltas and masks must have identical keys")
    payload = io.BytesIO()
    for name, delta in deltas.items():
        mask = masks[name]
        if mask.dtype != torch.bool or mask.shape != delta.shape:
            raise ValueError(f"Mask for {name!r} must be boolean and match its residual")
        payload.write(np.packbits(mask.numpy()).tobytes())
        survivors = delta[mask]
        if dtype == "int4":
            encoded, scale = int4_pack(survivors)
            payload.write(encoded)
            payload.write(np.float32(scale).tobytes())
        else:
            payload.write(survivors.to(torch.float16).numpy().tobytes())
    return payload.getvalue()


def size_rows(
    deltas: Mapping[str, torch.Tensor],
    ratios: Sequence[float],
    dtype: str,
) -> list[dict[str, float | int]]:
    """Return structured size measurements for all requested ratios."""
    parameter_count = sum(delta.numel() for delta in deltas.values())
    if parameter_count == 0:
        raise ValueError("Residual stream must contain at least one parameter")
    raw_fp32_bytes = parameter_count * 4
    raw_fp16_bytes = parameter_count * 2
    rows = []
    for ratio in ratios:
        masks = exact_residual_masks(deltas, ratio)
        payload = serialize_residual(deltas, masks, dtype)
        serialized_bytes = len(payload)
        compressed_bytes = len(zlib.compress(payload, 6))
        if serialized_bytes == 0 or compressed_bytes == 0:
            raise RuntimeError("Serialized residual payload must be non-empty")
        pruned_count = sum(int((~mask).sum().item()) for mask in masks.values())
        rows.append(
            {
                "prune_ratio": ratio,
                "parameter_count": parameter_count,
                "pruned_count": pruned_count,
                "raw_fp32_bytes": raw_fp32_bytes,
                "raw_fp16_bytes": raw_fp16_bytes,
                "serialized_bytes": serialized_bytes,
                "zlib_bytes": compressed_bytes,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-new", "--ckpt_new", dest="ckpt_new", type=Path, required=True)
    parser.add_argument("--ckpt-old", "--ckpt_old", dest="ckpt_old", type=Path, required=True)
    parser.add_argument("--ratios", default="0.5,0.7,0.9")
    parser.add_argument("--dtype", default="int4", choices=("int4", "fp16"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deltas = residuals(load_state(args.ckpt_new), load_state(args.ckpt_old))
    rows = size_rows(deltas, parse_ratios(args.ratios), args.dtype)
    parameter_count = rows[0]["parameter_count"]
    print(
        f"params covered: {parameter_count / 1e6:.1f}M | "
        f"raw fp32 {rows[0]['raw_fp32_bytes'] / 1e6:.1f} MB | "
        f"fp16 {rows[0]['raw_fp16_bytes'] / 1e6:.1f} MB"
    )
    for row in rows:
        raw32 = int(row["raw_fp32_bytes"])
        raw16 = int(row["raw_fp16_bytes"])
        serialized = int(row["serialized_bytes"])
        compressed = int(row["zlib_bytes"])
        print(
            f"p={row['prune_ratio']}: pruned={row['pruned_count']}/{parameter_count} | "
            f"serialized {serialized / 1e6:.1f} MB "
            f"({raw32 / serialized:.1f}x vs fp32, {raw16 / serialized:.1f}x vs fp16) | "
            f"+zlib {compressed / 1e6:.1f} MB "
            f"({raw32 / compressed:.1f}x vs fp32, {raw16 / compressed:.1f}x vs fp16)"
        )


if __name__ == "__main__":
    main()
