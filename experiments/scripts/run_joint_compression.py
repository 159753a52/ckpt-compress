"""
Table 2: 联合压缩对比（Pruning + Quantization）

对比不同方法在 pruning-only / quant-only / joint 模式下的压缩比和质量。
包含 ExCP 和 Inshrinkerator 的端到端管线对比。

运行示例:
    python experiments/scripts/run_joint_compression.py \
        --model gpt2-medium --dataset wikitext103 \
        --prune_ratios 0.2,0.3,0.4 --device cuda
"""

import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

import sys
from pathlib import Path
import argparse
import copy
import json
import time
import torch
import numpy as np
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lib.models import load_model
from experiments.lib.data import get_data_loaders, cache_batches
from experiments.lib.evaluation import evaluate
from experiments.lib.results import save_results, print_results_table
from experiments.lib.importance_compare.scoring import compute_scores_by_method
from dacp.pruning import apply_pruning, filter_prunable_params
from dacp.pruning.allocation import get_allocation_strategy
from dacp.pruning.importance import combine_scores_2d_with_protection, apply_magnitude_protection
from dacp.quantization import INT4Quantizer, KMeansQuantizer
from baselines.excp.excp import ExCPCompressor, ExCPConfig
from baselines.inshrinkerator.inshrinkerator import InshrinkeratorCompressor, InshrinkeratorConfig


PRUNING_METHODS = [
    {'label': 'Magnitude+Uniform',    'importance': 'magnitude',       'allocation': 'uniform'},
    {'label': 'First-order+Uniform',  'importance': 'first-order',     'allocation': 'uniform'},
    {'label': 'Ours (1D)',            'importance': 'second-order-hvp', 'allocation': 'gamma-adaptive'},
]

# 2D / protected 方法使用 second-order-hvp 的 allocation，但层内剪枝用组合得分
EXTRA_METHODS_2D = [
    {'label': 'Ours (2D)',            'importance': '_2d_combined',     'allocation': 'gamma-adaptive'},
    {'label': 'Ours (Protected)',     'importance': '_protected',       'allocation': 'gamma-adaptive'},
    {'label': 'Magnitude+DA',        'importance': '_mag_da',          'allocation': 'gamma-adaptive'},
]


def compute_compression_ratio(original_size, pruned_params, quant_bits=32):
    """计算压缩比。
    
    original_size: 原始参数总数
    pruned_params: 被剪枝的参数数
    quant_bits: 量化后的位数 (4 for INT4, 32 for no quant)
    """
    remaining = original_size - pruned_params
    compressed_bits = remaining * quant_bits + pruned_params * 1  # 1-bit mask
    original_bits = original_size * 32
    return original_bits / compressed_bits


def apply_quantization(model, quantizer):
    """对模型权重做 INT4 量化再反量化（模拟量化误差）。"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() >= 2:  # 只量化矩阵权重
                quantized, metadata = quantizer.quantize(param.data)
                dequantized = quantizer.dequantize(quantized, metadata)
                param.data.copy_(dequantized)


def run_single_config(model_init, scores, prune_ratio, method, cached_eval,
                      task_type, device, quantize, quant_type='int4', kmeans_clusters=256,
                      alloc_scores=None):
    """运行单个配置并返回结果。

    Args:
        alloc_scores: 若非 None，层间分配用此得分，层内剪枝用 scores。
    """
    model = copy.deepcopy(model_init).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    pruned_count = 0

    # Pruning
    if prune_ratio > 0 and scores is not None:
        allocator = get_allocation_strategy(method['allocation'])
        layer_ratios = allocator.allocate(alloc_scores if alloc_scores is not None else scores, prune_ratio)
        mask, pruned_count_val, actual = apply_pruning(
            model, scores, layer_ratios, device=device)
        pruned_count = int(actual * total_params)
    else:
        actual = 0.0

    # Quantization
    quant_bits = 32
    quant_label = 'None'
    if quantize:
        if quant_type == 'kmeans':
            quantizer = KMeansQuantizer(n_clusters=kmeans_clusters)
            quant_bits = int(np.ceil(np.log2(kmeans_clusters)))  # 256 clusters → 8-bit index
            quant_label = f'KMeans-{kmeans_clusters}'
        else:
            quantizer = INT4Quantizer()
            quant_bits = 4
            quant_label = 'INT4'
        apply_quantization(model, quantizer)

    # Evaluate
    metrics = evaluate(model, cached_eval, task_type, device)

    # Compression ratio
    cr = compute_compression_ratio(total_params, pruned_count, quant_bits)

    result = {
        'method': method['label'],
        'prune_ratio': prune_ratio,
        'actual_prune_ratio': actual,
        'quantize': quant_label,
        'compression_ratio': round(cr, 2),
    }
    result.update(metrics)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def _collect_gradients_simple(model, cached_train, task_type, num_batches, device):
    """收集梯度用于 Inshrinkerator 的敏感度计算。"""
    from experiments.lib.importance_compare.scoring import _make_loss_fn
    loss_fn = _make_loss_fn(task_type)
    model.train()
    model.zero_grad()
    count = 0
    for batch in cached_train[:num_batches]:
        batch_gpu = {k: v.to(device) for k, v in batch.items()}
        loss = loss_fn(model, batch_gpu)
        loss.backward()
        count += 1
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().cpu() / max(count, 1)
    model.zero_grad()
    return grads


def run_excp_e2e(model_init, cached_train, cached_eval, task_type, device,
                 target_prune_frac, num_steps):
    """ExCP 端到端：残差编码 + 联合剪枝 + K-means 量化。

    模拟单次 compress → decompress cycle。
    ExCP 的剪枝率由 alpha 超参间接控制，这里通过调整 alpha 近似目标稀疏度。
    """
    model = copy.deepcopy(model_init).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # 获取权重
    W_t = {n: p.data.detach().cpu() for n, p in model.named_parameters()}

    # 构造模拟的 optimizer states
    # 真实 ExCP 使用 Adam 的 exp_avg / exp_avg_sq
    # 这里用零/单位初始化近似首次 checkpoint 的情况
    O_t = {}
    for name, w in W_t.items():
        O_t[name] = {
            'exp_avg': torch.zeros_like(w),
            'exp_avg_sq': torch.ones_like(w) * 0.01,
        }

    # ExCP compress → decompress (首次 checkpoint，无前一检查点)
    config = ExCPConfig(alpha=5e-5, beta=2.0, p=0.0, n_bits=4)
    compressor = ExCPCompressor(config)
    compressed = compressor.compress(W_t, O_t, prev_W_hat=None)
    W_hat, _ = compressor.decompress(compressed, prev_W_hat=None)

    # 计算实际 sparsity 和 compression ratio
    pruned_count = 0
    for name in W_hat:
        pruned_count += int((W_hat[name] == 0).sum().item())
    actual_sparsity = pruned_count / total_params
    compressed_size = len(compressed)
    original_size = total_params * 4  # float32 = 4 bytes
    cr = original_size / compressed_size if compressed_size > 0 else 1.0

    # 加载重建权重到模型并评估
    model_eval = copy.deepcopy(model_init).to(device)
    with torch.no_grad():
        for name, param in model_eval.named_parameters():
            if name in W_hat:
                param.data.copy_(W_hat[name].to(device))
    metrics = evaluate(model_eval, cached_eval, task_type, device)

    result = {
        'method': 'ExCP (e2e)',
        'prune_ratio': target_prune_frac,
        'actual_prune_ratio': round(actual_sparsity, 4),
        'quantize': 'KMeans-16 (4-bit)',
        'compression_ratio': round(cr, 2),
    }
    result.update(metrics)

    del model, model_eval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_inshrinkerator_e2e(model_init, cached_train, cached_eval, task_type,
                           device, target_prune_frac, num_steps):
    """Inshrinkerator 端到端：三向分区 + 近似 K-means + 增量编码。

    模拟单次 compress → decompress cycle。
    """
    model = copy.deepcopy(model_init).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # 收集梯度
    grads = _collect_gradients_simple(model, cached_train, task_type,
                                      min(num_steps, 8), device)

    W_t = {n: p.data.detach().cpu() for n, p in model.named_parameters()}

    # Inshrinkerator compress → decompress (首次，无前一检查点)
    config = InshrinkeratorConfig(
        n_bins=16,
        protect_fraction=0.005,
        prune_fraction=target_prune_frac,
    )
    compressor = InshrinkeratorCompressor(config)
    compressed = compressor.compress(W_t, grads, prev_quantized=None)
    W_hat = compressor.decompress(compressed, prev_quantized=None)

    # 计算实际 sparsity 和 compression ratio
    pruned_count = 0
    for name in W_hat:
        pruned_count += int((W_hat[name] == 0).sum().item())
    actual_sparsity = pruned_count / total_params
    compressed_size = len(compressed)
    original_size = total_params * 4
    cr = original_size / compressed_size if compressed_size > 0 else 1.0

    # 加载重建权重到模型并评估
    model_eval = copy.deepcopy(model_init).to(device)
    with torch.no_grad():
        for name, param in model_eval.named_parameters():
            if name in W_hat:
                param.data.copy_(W_hat[name].to(device))
    metrics = evaluate(model_eval, cached_eval, task_type, device)

    result = {
        'method': 'Inshrinkerator (e2e)',
        'prune_ratio': target_prune_frac,
        'actual_prune_ratio': round(actual_sparsity, 4),
        'quantize': 'ApproxKMeans-16',
        'compression_ratio': round(cr, 2),
    }
    result.update(metrics)

    del model, model_eval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description='Table 2: 联合压缩对比')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--prune_ratios', type=str, default='0.2,0.3,0.4')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--hvp_batches', type=int, default=8)
    parser.add_argument('--eval_batches', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scoring_mode', type=str, default='all',
                        choices=['1d', '2d', 'protected', 'all'],
                        help='得分模式：1d=仅damage score，2d=magnitude+damage rank组合，'
                             'protected=magnitude保护+damage score，all=全部运行')
    parser.add_argument('--protection_ratio', type=float, default=0.001,
                        help='Protection比例（2d/protected模式），默认0.1%%')
    parser.add_argument('--output_dir', type=str,
                        default='results/paper_results/joint_compression')
    parser.add_argument('--skip_e2e', action='store_true',
                        help='跳过 ExCP 和 Inshrinkerator 端到端阶段')
    args = parser.parse_args()

    prune_ratios = [float(r) for r in args.prune_ratios.split(',')]

    print("=" * 70)
    print("Table 2: 联合压缩对比 (Pruning + Quantization)")
    print(f"模型: {args.model} | 数据集: {args.dataset}")
    print("=" * 70)

    model_init, model_family = load_model(args.model, pretrained=True,
                               checkpoint_path=args.checkpoint, device='cpu',
                               dataset_name=args.dataset)
    train_loader, val_loader, task_type = get_data_loaders(
        args.model, args.dataset, args.batch_size, args.seq_length)
    cached_train = cache_batches(train_loader, args.num_steps, task_type)
    cached_eval = cache_batches(val_loader, args.eval_batches, task_type)

    # Baseline: no compression
    print("\n[0] Baseline (no compression)...")
    model_base = copy.deepcopy(model_init).to(args.device)
    baseline = evaluate(model_base, cached_eval, task_type, args.device)
    print(f"  Baseline: {baseline}")
    del model_base

    # 计算所有 importance methods 的 scores
    print("\n[1] 计算 importance scores...")
    all_methods = list({m['importance'] for m in PRUNING_METHODS})
    model_s = copy.deepcopy(model_init).to(args.device)
    score_cache = compute_scores_by_method(
        model_s, cached_train, task_type,
        methods=all_methods, alpha=args.alpha,
        hvp_batches=args.hvp_batches,
        model_family=model_family,
    )
    del model_s
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 构建 2D / protected 变体得分
    score_cache_ext = dict(score_cache)  # 扩展缓存
    hvp_scores = score_cache.get('second-order-hvp')
    mag_scores = score_cache.get('magnitude')
    if hvp_scores is not None and mag_scores is not None:
        if args.scoring_mode in ('2d', 'all'):
            score_cache_ext['_2d_combined'] = combine_scores_2d_with_protection(
                mag_scores, hvp_scores, protection_ratio=args.protection_ratio,
                alpha=args.alpha)
            print(f"  2D combined scores computed (alpha={args.alpha}, protection={args.protection_ratio:.2%})")
        if args.scoring_mode in ('protected', 'all'):
            weights_cpu = {n: p.detach().cpu() for n, p in model_init.named_parameters()}
            prunable_w = filter_prunable_params(weights_cpu)
            score_cache_ext['_protected'] = apply_magnitude_protection(
                hvp_scores, prunable_w, protection_ratio=args.protection_ratio)
            del weights_cpu, prunable_w
            print(f"  Protected scores computed (protection={args.protection_ratio:.2%})")
        # Magnitude+DA: magnitude 层内剪枝 + HVP 层间分配
        if args.scoring_mode in ('2d', 'all'):
            score_cache_ext['_mag_da'] = mag_scores
            print("  Magnitude+DA scores ready (mag intra-layer, HVP allocation)")

    # 确定本次运行的方法集
    active_methods = list(PRUNING_METHODS)
    if args.scoring_mode in ('2d', 'all') and '_2d_combined' in score_cache_ext:
        active_methods.append(EXTRA_METHODS_2D[0])  # Ours (2D)
    if args.scoring_mode in ('protected', 'all') and '_protected' in score_cache_ext:
        active_methods.append(EXTRA_METHODS_2D[1])  # Ours (Protected)
    if args.scoring_mode in ('2d', 'all') and '_mag_da' in score_cache_ext:
        active_methods.append(EXTRA_METHODS_2D[2])  # Magnitude+DA

    # 释放不需要的 score 缓存以节省内存（cgroup ~32GB 限制）
    needed_keys = {m['importance'] for m in active_methods}
    # gamma-adaptive allocation 需要 second-order-hvp
    if any(m['importance'] in ('_2d_combined', '_protected', '_mag_da') for m in active_methods):
        needed_keys.add('second-order-hvp')
    for key in list(score_cache_ext.keys()):
        if key not in needed_keys:
            del score_cache_ext[key]
            print(f"  释放未使用的 score 缓存: {key}")
    # 也从原始 score_cache 中删除
    for key in list(score_cache.keys()):
        if key not in needed_keys:
            del score_cache[key]
    import gc; gc.collect()

    results = []

    # Quantization only (no pruning)
    print("\n[2] Quantization only...")
    for qt, ql in [('int4', 'INT4'), ('kmeans', 'KMeans-256')]:
        r = run_single_config(model_init, None, 0.0,
                              {'label': f'Quant-only ({ql})', 'allocation': 'uniform'},
                              cached_eval, task_type, args.device,
                              quantize=True, quant_type=qt)
        r['mode'] = 'quant-only'
        results.append(r)
        print(f"  {ql} only: CR={r['compression_ratio']:.1f}x | "
              + " | ".join(f"{k}={v:.4f}" for k, v in r.items()
                           if isinstance(v, float) and k != 'compression_ratio'))

    # For each method × ratio × (pruning-only, joint-INT4, joint-KMeans)
    print("\n[3] Pruning + Joint experiments...")
    for method in active_methods:
        imp_key = method['importance']
        # 2D/protected/mag_da 层间分配用原始 damage score，层内用各自得分
        if imp_key in ('_2d_combined', '_protected', '_mag_da'):
            scores = score_cache_ext[imp_key]
            alloc_scores = score_cache_ext.get('second-order-hvp', scores)
        else:
            scores = score_cache_ext[imp_key]
            alloc_scores = scores
        for ratio in prune_ratios:
            # Pruning only
            r_prune = run_single_config(
                model_init, scores, ratio, method,
                cached_eval, task_type, args.device, quantize=False,
                alloc_scores=alloc_scores if alloc_scores is not scores else None)
            r_prune['mode'] = 'prune-only'
            results.append(r_prune)

            # Joint: pruning + INT4
            r_int4 = run_single_config(
                model_init, scores, ratio, method,
                cached_eval, task_type, args.device,
                quantize=True, quant_type='int4',
                alloc_scores=alloc_scores if alloc_scores is not scores else None)
            r_int4['mode'] = 'joint'
            results.append(r_int4)

            # Joint: pruning + KMeans-256
            r_km = run_single_config(
                model_init, scores, ratio, method,
                cached_eval, task_type, args.device,
                quantize=True, quant_type='kmeans',
                alloc_scores=alloc_scores if alloc_scores is not scores else None)
            r_km['mode'] = 'joint'
            results.append(r_km)

            print(f"  {method['label']:30s} | prune={ratio:.0%} | "
                  f"P-only CR={r_prune['compression_ratio']:.1f}x | "
                  f"INT4 CR={r_int4['compression_ratio']:.1f}x | "
                  f"KMeans CR={r_km['compression_ratio']:.1f}x")

    # 增量保存 Phase 3 结果（防止后续阶段崩溃丢失数据）
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_dir),
                 f"joint_compression_{args.model}_{args.dataset}_phase3",
                 vars(args))
    print(f"  [Phase 3 保存完毕: {len(results)} 条结果]")

    # 释放 score cache 腾出内存给端到端 baseline
    del score_cache, score_cache_ext
    import gc; gc.collect()

    if not args.skip_e2e:
        # [4] ExCP end-to-end
        print("\n[4] ExCP 端到端压缩...")
        for prune_frac in prune_ratios:
            r_excp = run_excp_e2e(model_init, cached_train, cached_eval, task_type,
                                  args.device, prune_frac, args.num_steps)
            r_excp['mode'] = 'excp-e2e'
            results.append(r_excp)
            print(f"  ExCP (prune≈{prune_frac:.0%}) | CR={r_excp['compression_ratio']:.1f}x | "
                  + " | ".join(f"{k}={v:.4f}" for k, v in r_excp.items()
                               if isinstance(v, float) and k not in ('compression_ratio', 'prune_ratio', 'actual_prune_ratio')))

        # [5] Inshrinkerator end-to-end
        print("\n[5] Inshrinkerator 端到端压缩...")
        for prune_frac in prune_ratios:
            r_inshrink = run_inshrinkerator_e2e(
                model_init, cached_train, cached_eval, task_type,
                args.device, prune_frac, args.num_steps)
            r_inshrink['mode'] = 'inshrinkerator-e2e'
            results.append(r_inshrink)
            print(f"  Inshrinkerator (prune={prune_frac:.0%}) | CR={r_inshrink['compression_ratio']:.1f}x | "
                  + " | ".join(f"{k}={v:.4f}" for k, v in r_inshrink.items()
                               if isinstance(v, float) and k not in ('compression_ratio', 'prune_ratio', 'actual_prune_ratio')))
    else:
        print("\n[4-5] 跳过 ExCP / Inshrinkerator 端到端（--skip_e2e）")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_dir),
                 f"joint_compression_{args.model}_{args.dataset}",
                 vars(args))

    print("\n" + "=" * 70)
    print("Results:")
    print_results_table(results,
        ['method', 'prune_ratio', 'quantize', 'compression_ratio', 'loss', 'perplexity'])
    print("\n完成！")


if __name__ == '__main__':
    main()
