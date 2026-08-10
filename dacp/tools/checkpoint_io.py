"""Bitmask-based checkpoint compression serialization.

Format: for each parameter tensor:
  - Pruned layers: 1-bit mask (packed uint8) + dense non-zero values (fp16),
    or mask + uint8 cluster indices + fp16 codebook for quantized checkpoints
  - Non-pruned layers: raw values (fp16)

This yields an amortized cost of 1 + 16*(1-P) bits per parameter
for fp16 non-zero storage, where P is the sparsity ratio.
"""

import os
import numpy as np
import torch
from typing import Dict, Mapping


def _normalize_mask(
    name: str,
    param: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Validate one keep mask and return a detached CPU boolean tensor."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Mask for {name!r} must be a torch.Tensor")
    if mask.shape != param.shape:
        raise ValueError(
            f"Mask shape for {name!r} must match parameter shape: "
            f"{tuple(mask.shape)} != {tuple(param.shape)}"
        )
    mask_cpu = mask.detach().cpu()
    if mask_cpu.dtype != torch.bool:
        binary = (mask_cpu == 0) | (mask_cpu == 1)
        if not binary.all():
            raise ValueError(f"Mask for {name!r} must contain only 0 or 1")
    return mask_cpu.bool()


def save_compressed_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    path: str,
    use_fp16: bool = True,
) -> int:
    """Save checkpoint in bitmask format.

    Args:
        state_dict: model state_dict (original, before zeroing)
        masks: {layer_name: binary mask tensor} from apply_pruning
        path: output file path
        use_fp16: store non-zero values in fp16

    Returns:
        compressed file size in bytes
    """
    compressed = {}
    for name, param in state_dict.items():
        p = param.detach().cpu()
        if name in masks:
            mask = _normalize_mask(name, p, masks[name])
            values = p[mask]
            if use_fp16:
                values = values.half()
            # Pack mask bits into uint8 array
            packed_mask = np.packbits(mask.numpy().flatten().astype(np.uint8))
            compressed[name] = {
                'shape': list(p.shape),
                'mask': torch.from_numpy(packed_mask),
                'values': values,
                'numel': p.numel(),
            }
        else:
            compressed[name] = p.half() if use_fp16 else p

    torch.save(compressed, path)
    return os.path.getsize(path)


def _unpack_masked_entry(data: Mapping, dtype: torch.dtype) -> torch.Tensor:
    """Restore one masked entry from either dense values or quantized labels."""
    try:
        shape = tuple(data["shape"])
        numel = int(data["numel"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Masked checkpoint entry has invalid shape metadata") from exc
    if numel < 0 or int(np.prod(shape, dtype=np.int64)) != numel:
        raise ValueError("Masked checkpoint entry has inconsistent numel metadata")

    packed_mask = data.get("mask")
    if isinstance(packed_mask, torch.Tensor):
        packed_mask = packed_mask.detach().cpu().numpy()
    packed_mask = np.asarray(packed_mask, dtype=np.uint8)
    try:
        unpacked_mask = np.unpackbits(packed_mask)
    except ValueError as exc:
        raise ValueError("Masked checkpoint entry has an invalid packed mask") from exc
    if unpacked_mask.size < numel:
        raise ValueError("Masked checkpoint entry has a truncated packed mask")
    flat_mask = unpacked_mask[:numel].astype(bool)
    mask = torch.from_numpy(flat_mask)
    active_count = int(mask.sum().item())

    if "values" in data:
        values = data["values"]
    elif "indices" in data and "codebook" in data:
        indices = data["indices"]
        codebook = data["codebook"]
        if not isinstance(indices, torch.Tensor) or not isinstance(
            codebook, torch.Tensor
        ):
            raise ValueError(
                "Quantized checkpoint entries must store tensor indices and codebook"
            )
        indices = indices.detach().cpu().to(torch.long).flatten()
        codebook = codebook.detach().cpu().flatten()
        if indices.numel() != active_count:
            raise ValueError("Quantized checkpoint indices do not match the mask")
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= codebook.numel()
        ):
            raise ValueError("Quantized checkpoint index is outside the codebook")
        values = codebook[indices]
    else:
        raise ValueError("Masked checkpoint entry has neither values nor quantized data")

    if not isinstance(values, torch.Tensor) or values.numel() != active_count:
        raise ValueError("Masked checkpoint values do not match the mask")
    tensor = torch.zeros(shape, dtype=dtype)
    tensor.view(-1)[mask] = values.detach().cpu().flatten().to(dtype)
    return tensor


def load_compressed_checkpoint(
    path: str,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Load bitmask-compressed checkpoint back to dense state_dict.

    Both the legacy dense-value entries and the quantized
    ``indices``/``codebook`` entries produced by
    :func:`save_quantized_compressed_checkpoint` are supported.

    Args:
        path: compressed checkpoint file path
        dtype: target dtype for restored tensors

    Returns:
        Restored state_dict with zeros filled back in
    """
    compressed = torch.load(path, map_location='cpu')
    state_dict = {}
    for name, data in compressed.items():
        if isinstance(data, dict) and 'mask' in data:
            state_dict[name] = _unpack_masked_entry(data, dtype)
        else:
            state_dict[name] = data.to(dtype)
    return state_dict


def save_quantized_compressed_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    path: str,
    n_clusters: int = 256,
) -> int:
    """Save checkpoint with pruning bitmask + KMeans quantization.

    For pruned+quantized layers:
      - 1-bit mask (packed)
      - cluster indices (uint8 for k<=256)
      - codebook (k float16 centroids)

    Stored index cost: 8 * (1-P) bits per parameter plus the 1-bit mask.

    Returns:
        compressed file size in bytes
    """
    if (
        isinstance(n_clusters, bool)
        or not isinstance(n_clusters, int)
        or not 1 <= n_clusters <= 256
    ):
        raise ValueError(
            "n_clusters must be an integer in [1, 256] for uint8 indices, "
            f"got {n_clusters}"
        )
    compressed = {}

    for name, param in state_dict.items():
        p = param.detach().cpu()
        if name in masks:
            mask = _normalize_mask(name, p, masks[name])
            values = p[mask].numpy()
            packed_mask = np.packbits(mask.numpy().flatten().astype(np.uint8))

            if len(values) > n_clusters:
                # Quantize non-zero values
                from sklearn.cluster import MiniBatchKMeans
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    batch_size=min(10000, len(values)),
                    max_iter=50,
                    random_state=42,
                )
                kmeans.fit(values.reshape(-1, 1))
                centroids = kmeans.cluster_centers_.flatten().astype(np.float16)
                indices = kmeans.predict(values.reshape(-1, 1)).astype(np.uint8)
                compressed[name] = {
                    'shape': list(p.shape),
                    'mask': torch.from_numpy(packed_mask),
                    'indices': torch.from_numpy(indices),
                    'codebook': torch.from_numpy(centroids),
                    'numel': p.numel(),
                }
            else:
                compressed[name] = {
                    'shape': list(p.shape),
                    'mask': torch.from_numpy(packed_mask),
                    'values': torch.from_numpy(values).half(),
                    'numel': p.numel(),
                }
        else:
            compressed[name] = p.half()

    torch.save(compressed, path)
    return os.path.getsize(path)


def get_size_breakdown(
    state_dict: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    use_fp16: bool = True,
) -> dict:
    """Calculate theoretical size breakdown without writing files.

    Returns dict with:
        original_bytes, mask_bytes, values_bytes, total_compressed_bytes,
        compression_ratio, sparsity
    """
    total_original = 0
    total_mask_bytes = 0
    total_value_bits = 0
    total_pruned = 0
    total_params = 0

    for name, param in state_dict.items():
        n = param.numel()
        total_original += n * param.element_size()
        value_bits = 16 if use_fp16 else param.element_size() * 8

        if name in masks:
            mask = _normalize_mask(name, param, masks[name])
            nnz = int(mask.sum().item())
            total_mask_bytes += (n + 7) // 8  # np.packbits pads each tensor separately
            total_value_bits += nnz * value_bits
            total_pruned += (n - nnz)
            total_params += n
        else:
            total_value_bits += n * value_bits

    mask_bytes = total_mask_bytes
    value_bytes = total_value_bits // 8
    total_compressed = mask_bytes + value_bytes

    return {
        'original_bytes': total_original,
        'mask_bytes': mask_bytes,
        'values_bytes': value_bytes,
        'total_compressed_bytes': total_compressed,
        'compression_ratio': total_original / total_compressed if total_compressed > 0 else float('inf'),
        'sparsity': total_pruned / total_params if total_params > 0 else 0,
        'total_params': total_params,
        'pruned_params': total_pruned,
    }
