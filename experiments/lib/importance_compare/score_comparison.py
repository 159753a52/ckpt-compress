"""Score comparison utilities for global vs block-wise HVP analysis.

提供 Spearman/Pearson 相关系数、relative L2 error、pruning mask IoU 等
对比度量，用于验证 block-diagonal HVP 近似的精度。
"""

import torch
import numpy as np
from typing import Dict
from scipy import stats as scipy_stats

# 超过此阈值时使用随机采样（全局 + 逐层通用）
_CORR_SAMPLE_THRESHOLD = 200_000
_CORR_SAMPLE_SIZE = 200_000


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
    common_keys = sorted(set(scores_ref) & set(scores_approx))
    result = {}
    all_ref, all_approx = [], []

    for name in common_keys:
        ref = scores_ref[name].flatten().float()
        approx = scores_approx[name].flatten().float()
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
    common_keys = sorted(set(scores_ref) & set(scores_approx))
    n_keys = len(common_keys)
    result = {}
    all_ref, all_approx = [], []

    rng = np.random.RandomState(42)

    for i, name in enumerate(common_keys):
        print(f"\r  [Correlation] {i+1}/{n_keys}", end="", flush=True)
        ref = scores_ref[name].flatten().float().numpy()
        approx = scores_approx[name].flatten().float().numpy()

        if len(ref) < 3 or np.std(ref) < 1e-12 or np.std(approx) < 1e-12:
            result[name] = {'spearman': 1.0, 'pearson': 1.0}
        else:
            # 逐层也采样，避免大参数层（如 c_attn weight ~3M）拖慢整体
            if len(ref) > _CORR_SAMPLE_THRESHOLD:
                idx = rng.choice(len(ref), _CORR_SAMPLE_SIZE, replace=False)
                ref_s, approx_s = ref[idx], approx[idx]
            else:
                ref_s, approx_s = ref, approx
            sp = scipy_stats.spearmanr(ref_s, approx_s).statistic
            pr = scipy_stats.pearsonr(ref_s, approx_s).statistic
            result[name] = {'spearman': float(sp), 'pearson': float(pr)}

        all_ref.append(ref)
        all_approx.append(approx)

    print()  # newline after progress
    if all_ref:
        print("  [Correlation] computing global...", flush=True)
        cat_ref = np.concatenate(all_ref)
        cat_approx = np.concatenate(all_approx)
        if np.std(cat_ref) < 1e-12 or np.std(cat_approx) < 1e-12:
            result['__global__'] = {'spearman': 1.0, 'pearson': 1.0}
        else:
            if len(cat_ref) > _CORR_SAMPLE_THRESHOLD:
                idx = rng.choice(len(cat_ref), _CORR_SAMPLE_SIZE, replace=False)
                cat_ref_s, cat_approx_s = cat_ref[idx], cat_approx[idx]
            else:
                cat_ref_s, cat_approx_s = cat_ref, cat_approx
            sp = scipy_stats.spearmanr(cat_ref_s, cat_approx_s).statistic
            pr = scipy_stats.pearsonr(cat_ref_s, cat_approx_s).statistic
            result['__global__'] = {'spearman': float(sp), 'pearson': float(pr)}

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
    def _global_threshold(scores, ratio):
        all_scores = torch.cat([s.flatten().float() for s in scores.values()])
        k = int(len(all_scores) * ratio)
        if k <= 0:
            return -float('inf')
        threshold = torch.kthvalue(all_scores, k).values.item()
        return threshold

    common_keys = sorted(set(scores_ref) & set(scores_approx))
    thresh_ref = _global_threshold({k: scores_ref[k] for k in common_keys}, prune_ratio)
    thresh_approx = _global_threshold({k: scores_approx[k] for k in common_keys}, prune_ratio)

    total_intersection = 0
    total_union = 0
    total_agree = 0
    total_params = 0

    for name in common_keys:
        ref = scores_ref[name].flatten().float()
        approx = scores_approx[name].flatten().float()
        mask_ref = ref <= thresh_ref
        mask_approx = approx <= thresh_approx
        total_intersection += (mask_ref & mask_approx).sum().item()
        total_union += (mask_ref | mask_approx).sum().item()
        total_agree += (mask_ref == mask_approx).sum().item()
        total_params += ref.numel()

    iou = total_intersection / max(total_union, 1)
    agreement = total_agree / max(total_params, 1)

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

    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")

    cfg = model_configs[model_name]
    d = cfg['d']
    T = seq_length

    # ViT: T = (image_size / patch_size)² + 1 (CLS token)
    if model_name == 'vit-l-32':
        T = (224 // 32) ** 2 + 1  # = 50
    elif model_name == 'vit-b-16':
        T = (224 // 16) ** 2 + 1  # = 197

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
