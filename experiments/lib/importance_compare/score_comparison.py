"""Score comparison utilities for global vs block-wise HVP analysis.

提供 Spearman/Pearson 相关系数、relative L2 error、pruning mask IoU 等
对比度量，用于验证 block-diagonal HVP 近似的精度。
"""

import torch
import numpy as np
from typing import Dict, List, Mapping, Tuple
from scipy import stats as scipy_stats

# 超过此阈值时使用随机采样（全局 + 逐层通用）
_CORR_SAMPLE_THRESHOLD = 200_000
_CORR_SAMPLE_SIZE = 200_000


def _aligned_score_pairs(
    scores_ref: Mapping[str, torch.Tensor],
    scores_approx: Mapping[str, torch.Tensor],
) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Align score tensors and validate the shared comparison contract."""
    common_keys = sorted(set(scores_ref) & set(scores_approx))
    if not common_keys:
        raise ValueError("Score mappings must share at least one parameter")

    pairs = []
    for name in common_keys:
        ref_tensor = scores_ref[name].detach().float().cpu()
        approx_tensor = scores_approx[name].detach().float().cpu()
        if ref_tensor.shape != approx_tensor.shape:
            raise ValueError(
                f"Score shapes for {name!r} must match: "
                f"{tuple(ref_tensor.shape)} != {tuple(approx_tensor.shape)}"
            )
        ref = ref_tensor.flatten()
        approx = approx_tensor.flatten()
        if ref.numel() == 0:
            raise ValueError(f"Score tensor for {name!r} must not be empty")
        if not torch.isfinite(ref).all() or not torch.isfinite(approx).all():
            raise ValueError(f"Score tensors for {name!r} must contain finite values")
        pairs.append((name, ref, approx))
    return pairs


def _validate_prune_ratio(prune_ratio: float) -> float:
    if isinstance(prune_ratio, bool):
        raise ValueError(f"prune_ratio must be a finite number in [0, 1], got {prune_ratio}")
    try:
        ratio = float(prune_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"prune_ratio must be a finite number in [0, 1], got {prune_ratio}"
        ) from exc
    if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(
            f"prune_ratio must be a finite number in [0, 1], got {prune_ratio}"
        )
    return ratio


def _exact_pruned_mask(scores: torch.Tensor, prune_count: int) -> torch.Tensor:
    """Return a stable boolean mask with exactly ``prune_count`` true entries."""
    if not 0 <= prune_count <= scores.numel():
        raise ValueError(
            f"prune_count must be in [0, {scores.numel()}], got {prune_count}"
        )
    mask = torch.zeros(scores.numel(), dtype=torch.bool)
    if prune_count:
        order = torch.argsort(scores, stable=True)
        mask[order[:prune_count]] = True
    return mask


def _correlation_pair(ref: np.ndarray, approx: np.ndarray) -> Dict[str, float]:
    """Compute finite correlations, defining constant-score edge cases."""
    ref_constant = len(ref) == 0 or np.allclose(ref, ref[0], atol=1e-12, rtol=0.0)
    approx_constant = len(approx) == 0 or np.allclose(
        approx, approx[0], atol=1e-12, rtol=0.0
    )
    if ref_constant or approx_constant:
        identical_constants = ref_constant and approx_constant and np.array_equal(
            ref, approx
        )
        value = 1.0 if identical_constants else 0.0
        return {"spearman": value, "pearson": value}
    if len(ref) < 3:
        return {"spearman": 0.0, "pearson": 0.0}
    spearman = scipy_stats.spearmanr(ref, approx).statistic
    pearson = scipy_stats.pearsonr(ref, approx).statistic
    return {
        "spearman": float(spearman) if np.isfinite(spearman) else 0.0,
        "pearson": float(pearson) if np.isfinite(pearson) else 0.0,
    }


def compute_relative_l2_error(
    scores_ref: Dict[str, torch.Tensor],
    scores_approx: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """计算每层及全局的 relative L2 error。

    error_ℓ = ||s_ref - s_approx||₂ / ||s_ref||₂

    参数:
        scores_ref: 参考得分（global HVP）
        scores_approx: 近似得分（block-wise HVP）

    返回:
        {'layer_name': error, ..., '__global__': global_error}
    """
    pairs = _aligned_score_pairs(scores_ref, scores_approx)
    result = {}
    all_ref, all_approx = [], []

    for name, ref, approx in pairs:
        norm_ref = torch.norm(ref).item()
        if norm_ref < 1e-12:
            result[name] = 0.0
        else:
            result[name] = torch.norm(ref - approx).item() / norm_ref
        all_ref.append(ref)
        all_approx.append(approx)

    if all_ref:
        cat_ref = torch.cat(all_ref)
        cat_approx = torch.cat(all_approx)
        norm_global = torch.norm(cat_ref).item()
        result['__global__'] = torch.norm(cat_ref - cat_approx).item() / max(norm_global, 1e-12)

    return result


def compute_rank_correlation(
    scores_ref: Dict[str, torch.Tensor],
    scores_approx: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    """计算每层及全局的 Spearman/Pearson 相关系数。

    参数:
        scores_ref: 参考得分（global HVP）
        scores_approx: 近似得分（block-wise HVP）

    返回:
        {'layer_name': {'spearman': r, 'pearson': r}, ..., '__global__': {...}}
    """
    pairs = _aligned_score_pairs(scores_ref, scores_approx)
    n_keys = len(pairs)
    result = {}
    all_ref, all_approx = [], []

    rng = np.random.RandomState(42)

    for i, (name, ref_tensor, approx_tensor) in enumerate(pairs):
        print(f"\r  [Correlation] {i+1}/{n_keys}", end="", flush=True)
        ref = ref_tensor.numpy()
        approx = approx_tensor.numpy()

        # 逐层也采样，避免大参数层（如 c_attn weight ~3M）拖慢整体
        if len(ref) > _CORR_SAMPLE_THRESHOLD:
            idx = rng.choice(len(ref), _CORR_SAMPLE_SIZE, replace=False)
            ref_s, approx_s = ref[idx], approx[idx]
        else:
            ref_s, approx_s = ref, approx
        result[name] = _correlation_pair(ref_s, approx_s)

        all_ref.append(ref)
        all_approx.append(approx)

    print()  # newline after progress
    if all_ref:
        print("  [Correlation] computing global...", flush=True)
        cat_ref = np.concatenate(all_ref)
        cat_approx = np.concatenate(all_approx)
        if len(cat_ref) > _CORR_SAMPLE_THRESHOLD:
            idx = rng.choice(len(cat_ref), _CORR_SAMPLE_SIZE, replace=False)
            cat_ref_s, cat_approx_s = cat_ref[idx], cat_approx[idx]
        else:
            cat_ref_s, cat_approx_s = cat_ref, cat_approx
        result['__global__'] = _correlation_pair(cat_ref_s, cat_approx_s)

    return result


def compute_mask_iou(
    scores_ref: Dict[str, torch.Tensor],
    scores_approx: Dict[str, torch.Tensor],
    prune_ratio: float,
) -> Dict[str, float]:
    """计算给定剪枝率下的 pruning mask IoU。

    对两组 score 分别以全局阈值做 pruning 决策，计算 mask 交并比。

    参数:
        scores_ref: 参考得分
        scores_approx: 近似得分
        prune_ratio: 全局剪枝率

    返回:
        {'__global__': iou, '__agreement__': fraction}
    """
    prune_ratio = _validate_prune_ratio(prune_ratio)
    pairs = _aligned_score_pairs(scores_ref, scores_approx)
    total_params = sum(ref.numel() for _, ref, _ in pairs)
    prune_count = int(total_params * prune_ratio)
    ref_flat = torch.cat([ref for _, ref, _ in pairs])
    approx_flat = torch.cat([approx for _, _, approx in pairs])
    ref_pruned = _exact_pruned_mask(ref_flat, prune_count)
    approx_pruned = _exact_pruned_mask(approx_flat, prune_count)

    total_intersection = 0
    total_union = 0
    total_agree = 0
    offset = 0
    for name, ref, approx in pairs:
        end = offset + ref.numel()
        mask_ref = ref_pruned[offset:end]
        mask_approx = approx_pruned[offset:end]
        total_intersection += (mask_ref & mask_approx).sum().item()
        total_union += (mask_ref | mask_approx).sum().item()
        total_agree += (mask_ref == mask_approx).sum().item()
        offset = end

    iou = total_intersection / total_union if total_union else 1.0
    agreement = total_agree / total_params

    return {'__global__': iou, '__agreement__': agreement}


def compute_gamma(
    model_name: str,
    seq_length: int,
) -> Dict[str, float]:
    """计算模型的瓶颈比 γ = dT / n_ℓ。

    参数:
        model_name: 模型名称
        seq_length: 序列长度 T

    返回:
        {'d': hidden_dim, 'T': seq_length, 'n_ell': params_per_layer, 'gamma': ratio}
    """
    # 典型 Transformer 参数: n_ℓ ≈ 12d² (4d² attention + 8d² MLP)
    model_configs = {
        'gpt2-small':  {'d': 768,  'n_layers': 12},
        'gpt2-medium': {'d': 1024, 'n_layers': 24},
        'bert-base':   {'d': 768,  'n_layers': 12},
        'bert-large':  {'d': 1024, 'n_layers': 24},
        'pythia-410m':  {'d': 1024, 'n_layers': 24},
        'vit-l-32':    {'d': 1024, 'n_layers': 24},
        'vit-b-16':    {'d': 768,  'n_layers': 12},
    }

    if (
        isinstance(seq_length, bool)
        or not isinstance(seq_length, int)
        or seq_length < 1
    ):
        raise ValueError(f"seq_length must be a positive integer, got {seq_length}")
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")

    cfg = model_configs[model_name]
    d = cfg['d']
    T = seq_length

    # ViT: T = (image_size / patch_size)² + 1 (CLS token).  Keep these
    # values aligned with the model constructors in experiments.lib.models.
    vit_specs = {
        'vit-l-32': (384, 32),
        'vit-b-16': (224, 16),
    }
    if model_name in vit_specs:
        image_size, patch_size = vit_specs[model_name]
        T = (image_size // patch_size) ** 2 + 1

    n_ell = 12 * d * d  # 标准 Transformer 层参数量近似
    gamma = d * T / n_ell  # = T / (12d)

    return {
        'd': d,
        'T': T,
        'n_ell': n_ell,
        'gamma': gamma,
        'gamma_pct': gamma * 100,
    }


def summarize_comparison(
    l2_errors: Dict[str, float],
    correlations: Dict[str, Dict[str, float]],
    mask_ious: Dict[str, Dict[str, float]],
    gamma_info: Dict[str, float],
) -> str:
    """生成对比总结文本。"""
    lines = [
        f"=== Block-wise HVP Approximation Analysis ===",
        f"",
        f"Model config: d={gamma_info['d']}, T={gamma_info['T']}, "
        f"n_ℓ≈{gamma_info['n_ell']}, γ={gamma_info['gamma_pct']:.2f}%",
        f"",
        f"--- Global Metrics ---",
        f"Relative L2 Error:  {l2_errors.get('__global__', 0):.6f}",
        f"Spearman ρ:         {correlations.get('__global__', {}).get('spearman', 0):.6f}",
        f"Pearson r:          {correlations.get('__global__', {}).get('pearson', 0):.6f}",
    ]

    for ratio_key, iou_data in mask_ious.items():
        lines.append(
            f"Mask IoU @{ratio_key}: {iou_data['__global__']:.4f} "
            f"(agreement: {iou_data['__agreement__']:.4f})"
        )

    # 按层的 L2 error (top-5 worst)
    layer_errors = {k: v for k, v in l2_errors.items() if k != '__global__'}
    if layer_errors:
        lines.append("")
        lines.append("--- Top-5 Layers with Largest L2 Error ---")
        for name, err in sorted(layer_errors.items(), key=lambda x: -x[1])[:5]:
            sp = correlations.get(name, {}).get('spearman', 0)
            lines.append(f"  {name}: L2={err:.6f}, Spearman={sp:.4f}")

    return "\n".join(lines)
