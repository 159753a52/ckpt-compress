"""Bitmask-based checkpoint compression serialization.

Format: for each parameter tensor:
  - Pruned layers: 1-bit mask (packed uint8) + dense non-zero values (fp16)
  - Non-pruned layers: raw values (fp16)

This yields an amortized cost of 1 + 16*(1-P) bits per parameter
for fp16 non-zero storage, where P is the sparsity ratio.
"""

import os
import numpy as np
import torch
from typing import Dict


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
            mask = masks[name].bool().cpu()
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


def load_compressed_checkpoint(
    path: str,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Load bitmask-compressed checkpoint back to dense state_dict.

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
            shape = data['shape']
            numel = data['numel']
            packed = data['mask'].numpy()
            flat_mask = np.unpackbits(packed)[:numel].astype(bool)
            mask = torch.from_numpy(flat_mask).reshape(shape)
            tensor = torch.zeros(shape, dtype=dtype)
            tensor[mask] = data['values'].to(dtype)
            state_dict[name] = tensor
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

    Cost: 1 + ceil(log2(k)) * (1-P) bits per param.

    Returns:
        compressed file size in bytes
    """
    from ..quantization.kmeans import KMeansQuantizer

    quantizer = KMeansQuantizer(n_clusters=n_clusters)
    index_bits = int(np.ceil(np.log2(n_clusters)))
    compressed = {}

    for name, param in state_dict.items():
        p = param.detach().cpu()
        if name in masks:
            mask = masks[name].bool().cpu()
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

    val_bits = 16 if use_fp16 else 32

    for name, param in state_dict.items():
        n = param.numel()
        total_original += n * 4  # fp32 original

        if name in masks:
            mask = masks[name]
            nnz = int(mask.sum().item())
            total_mask_bytes += (n + 7) // 8  # np.packbits pads each tensor separately
            total_value_bits += nnz * val_bits
            total_pruned += (n - nnz)
            total_params += n
        else:
            total_value_bits += n * val_bits

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
