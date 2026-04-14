"""Fault-tolerant training simulation.

Simulate K compression-restoration cycles during training.
Measure end-to-end quality degradation from repeated lossy restoration.

Methods:
  - none: no compression (oracle baseline)
  - magnitude+uniform: magnitude pruning + uniform allocation
  - ours-2d: 2D scoring (mag+damage) + gamma-adaptive allocation
  - ours-2d+kmeans16: 2D scoring + pruning + KMeans-16 quantization

Usage:
    python experiments/scripts/run_fault_tolerant_training.py \
        --model gpt2-medium --dataset wikitext103 \
        --total_steps 1000 --num_recoveries 10 \
        --prune_ratio 0.3 --seq_length 128 --batch_size 2 \
        --device cuda
"""

import gc
import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lib.models import load_model
from experiments.lib.data import get_data_loaders, cache_batches
from experiments.lib.evaluation import evaluate
from experiments.lib.results import save_results
from dacp.pruning import Pruner, filter_prunable_params
from dacp.pruning.importance import (
    combine_scores_2d_with_protection,
    get_importance_scorer,
)
from dacp.quantization.kmeans import KMeansQuantizer
from dacp.quantization.int4 import INT4Quantizer


METHODS = [
    'none',
    'magnitude+uniform',
    'ours-2d',
    'ours-2d+kmeans16',
]


def train_one_step(model, optimizer, batch, task_type, device):
    """训练一步，返回 loss。"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    if task_type == 'lm':
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    elif task_type == 'cls':
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attn = batch.get('attention_mask')
        if attn is not None:
            attn = attn.to(device)
        outputs = model(input_ids, attention_mask=attn)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        loss = criterion(logits, labels)
    elif task_type == 'cv':
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(images)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        loss = criterion(logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def collect_gradients_inline(model, optimizer, train_loader, device, num_steps, task_type):
    """在训练过程中收集梯度（用于 first-order importance scoring）。"""
    criterion = nn.CrossEntropyLoss()
    accumulated_grads = defaultdict(lambda: 0)
    
    data_iter = iter(train_loader)
    model.train()
    
    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        optimizer.zero_grad()
        
        if task_type == 'lm':
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        elif task_type == 'cls':
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attn = batch.get('attention_mask')
            if attn is not None:
                attn = attn.to(device)
            outputs = model(input_ids, attention_mask=attn)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)
        elif task_type == 'cv':
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)
        
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                accumulated_grads[name] = accumulated_grads[name] + param.grad.detach().cpu()
        
        optimizer.step()
    
    for name in accumulated_grads:
        accumulated_grads[name] /= num_steps
    
    weights = {name: p.detach().cpu() for name, p in model.named_parameters()}
    return weights, dict(accumulated_grads)


def apply_pruning_to_model(model, scores, layer_ratios, device):
    """Apply pruning mask to model weights only (no optimizer modification).
    
    Matches Inshrinkerator behavior: only compress weights, leave optimizer intact.
    """
    model_params = dict(model.named_parameters())
    masks = {}
    
    for name, score in scores.items():
        if name not in model_params or name not in layer_ratios:
            continue
        
        ratio = layer_ratios[name]
        flat = score.flatten()
        k = int(len(flat) * ratio)
        if k == 0:
            masks[name] = torch.ones_like(score)
            continue
        
        threshold = torch.kthvalue(flat, k).values
        mask = (score > threshold).float()
        masks[name] = mask
        
        param = model_params[name]
        param.data.mul_(mask.to(device))
    
    return masks


# 不量化 embedding 和 layernorm（只量化 weight matrices）
_SKIP_QUANTIZE = ['embed', 'wte', 'wpe', 'ln_', 'LayerNorm', 'layernorm']

def apply_quantize_dequantize(model, masks, quantizer, device):
    """Apply quantize-then-dequantize to surviving (non-pruned) parameters.
    
    This simulates the lossy compression from quantization by replacing
    the actual values with their quantized-then-dequantized approximation.
    
    Uses mask-aware quantization: pruned positions (mask=0) are excluded
    from K-means clustering, ensuring all centroids represent meaningful values.
    """
    model_params = dict(model.named_parameters())
    
    for name, param in model_params.items():
        if param.dim() < 2:
            continue
        if any(pat in name for pat in _SKIP_QUANTIZE):
            continue
        # Pass pruning mask so quantizer excludes zeros from clustering
        layer_mask = masks.get(name, None)
        quantized, metadata = quantizer.quantize(param.data, mask=layer_mask)
        dequantized = quantizer.dequantize(quantized, metadata)
        param.data.copy_(dequantized.to(device))
        # Re-apply mask to ensure pruned positions stay zero
        if name in masks:
            param.data.mul_(masks[name].to(device))


def compute_scores_for_method(method_name, model, optimizer, train_loader,
                               device, num_steps, task_type, args):
    """Compute importance scores based on method name."""
    weights, grads = collect_gradients_inline(
        model, optimizer, train_loader, device, num_steps, task_type)
    
    prunable_w = filter_prunable_params(weights)
    prunable_g = {k: grads[k] for k in prunable_w if k in grads}
    
    if method_name.startswith('magnitude'):
        # Magnitude scoring + uniform allocation
        pruner = Pruner(importance='magnitude', allocation='uniform')
        scores = pruner.compute_scores(prunable_w, prunable_g)
        layer_ratios = pruner.compute_layer_ratios(scores, args.prune_ratio)
        return scores, layer_ratios
    
    elif method_name.startswith('ours-2d'):
        # 2D scoring: need HVP for damage scores
        # Compute magnitude scores
        mag_scorer = get_importance_scorer('magnitude')
        mag_scores = mag_scorer.score(prunable_w, prunable_g)
        
        # Compute first-order damage scores (lightweight, no HVP needed for FT experiment)
        # Use |g * theta| as damage proxy to avoid expensive HVP in every recovery cycle
        damage_scores = {}
        for name in prunable_w:
            w = prunable_w[name]
            g = prunable_g.get(name, torch.zeros_like(w))
            damage_scores[name] = (g * w).abs()
        
        # 2D combination
        combined = combine_scores_2d_with_protection(
            mag_scores, damage_scores,
            protection_ratio=args.protection_ratio,
            alpha=args.alpha,
        )
        
        # Gamma-adaptive allocation using damage scores
        pruner = Pruner(importance='magnitude', allocation='gamma-adaptive')
        alloc_scores = damage_scores
        layer_ratios = pruner.compute_layer_ratios(alloc_scores, args.prune_ratio)
        
        # 释放中间张量
        del weights, grads, prunable_w, prunable_g, mag_scores, damage_scores
        gc.collect()
        
        return combined, layer_ratios
    
    else:
        raise ValueError(f"Unknown method: {method_name}")


def run_method(method_name, args, train_loader, val_batches, task_type):
    """Run a complete training process for one method."""
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")
    
    model, _ = load_model(args.model, pretrained=True, 
                           checkpoint_path=args.checkpoint, device=args.device,
                           dataset_name=args.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Setup quantizer if needed
    quantizer = None
    if '+kmeans' in method_name:
        n_clusters = 16  # default
        if 'kmeans4' in method_name:
            n_clusters = 4
        elif 'kmeans16' in method_name:
            n_clusters = 16
        elif 'kmeans256' in method_name:
            n_clusters = 256
        quantizer = KMeansQuantizer(n_clusters=n_clusters)
    elif '+int4' in method_name:
        quantizer = INT4Quantizer()
    
    steps_per_segment = args.total_steps // (args.num_recoveries + 1)
    loss_history = []
    global_step = 0
    data_iter = iter(train_loader)
    
    for seg in range(args.num_recoveries + 1):
        print(f"\n--- Segment {seg+1}/{args.num_recoveries+1} ---")
        
        for step in range(steps_per_segment):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            loss = train_one_step(model, optimizer, batch, task_type, args.device)
            global_step += 1
            
            if step % args.eval_interval == 0:
                metrics = evaluate(model, val_batches, task_type, args.device)
                ppl = metrics.get('perplexity', 0)
                acc = metrics.get('accuracy', 0)
                loss_history.append({
                    'step': global_step, 'loss': metrics['loss'],
                    'perplexity': ppl, 'accuracy': acc,
                    'segment': seg,
                })
        
        # Compress at recovery point (skip last segment)
        if seg < args.num_recoveries and method_name != 'none':
            print(f"  [Recovery {seg+1}] Compressing checkpoint...")
            
            # Compute scores
            scores, layer_ratios = compute_scores_for_method(
                method_name, model, optimizer, train_loader,
                args.device, args.num_importance_steps, task_type, args)
            
            # Apply pruning (weights only, no optimizer modification)
            masks = apply_pruning_to_model(model, scores, layer_ratios, args.device)
            
            # Apply quantization if configured
            if quantizer is not None:
                apply_quantize_dequantize(model, masks, quantizer, args.device)
            
            # 释放 scoring 中间产物
            del scores, layer_ratios
            gc.collect()
            
            metrics = evaluate(model, val_batches, task_type, args.device)
            ppl_str = f"PPL={metrics.get('perplexity', 0):.2f}" if 'perplexity' in metrics else f"loss={metrics['loss']:.4f}"
            print(f"  After compression: {ppl_str}")
    
    # Final evaluation
    final_metrics = evaluate(model, val_batches, task_type, args.device)
    print(f"\nFinal: {final_metrics}")
    
    return loss_history, final_metrics


def plot_results(all_results, args):
    """绘制 loss 曲线。"""
    plt.figure(figsize=(12, 6))
    colors = {'none': 'black', 'magnitude+uniform': 'blue',
              'ours-2d': 'red', 'ours-2d+kmeans16': 'darkred',
              'magnitude+weibull-adaptive': 'green',
              'first-order+uniform': 'gray',
              'first-order+weibull-adaptive': 'orange'}
    
    for method, history in all_results.items():
        steps = [h['step'] for h in history]
        metric_key = 'perplexity' if 'perplexity' in history[0] and history[0]['perplexity'] > 0 else 'loss'
        values = [h[metric_key] for h in history]
        plt.plot(steps, values, label=method, color=colors.get(method, 'purple'), linewidth=2)
    
    steps_per_seg = args.total_steps // (args.num_recoveries + 1)
    for i in range(1, args.num_recoveries + 1):
        plt.axvline(x=i * steps_per_seg, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Validation PPL / Loss')
    plt.title(f'Fault-Tolerant Training: {args.model} on {args.dataset}\n'
              f'(K={args.num_recoveries} recoveries, prune={args.prune_ratio})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f'ft_{args.model}_{args.dataset}_K{args.num_recoveries}.png', dpi=300)
    plt.savefig(out_dir / f'ft_{args.model}_{args.dataset}_K{args.num_recoveries}.pdf')
    plt.close()
    print(f"图片已保存: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Fault-tolerant training simulation')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--total_steps', type=int, default=1000)
    parser.add_argument('--num_recoveries', type=int, default=10)
    parser.add_argument('--prune_ratio', type=float, default=0.3)
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated method list')
    parser.add_argument('--num_importance_steps', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seq_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='2D scoring weight for magnitude (0-1)')
    parser.add_argument('--protection_ratio', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results/paper_results/fault_tolerant')
    args = parser.parse_args()
    
    methods = [m.strip() for m in args.methods.split(',')] if args.methods else METHODS
    
    print("=" * 60)
    print("Fault-Tolerant Training Simulation")
    print(f"Model: {args.model} | Dataset: {args.dataset}")
    print(f"Steps: {args.total_steps} | Recoveries: {args.num_recoveries}")
    print(f"Prune ratio: {args.prune_ratio} | Methods: {methods}")
    print(f"Alpha: {args.alpha} | Protection: {args.protection_ratio}")
    print("=" * 60)
    
    train_loader, val_loader, task_type = get_data_loaders(
        args.model, args.dataset, args.batch_size, args.seq_length,
        data_dir=args.data_dir or './data')
    
    val_batches = cache_batches(val_loader, 50, task_type)
    
    # Baseline evaluation
    model_tmp, _ = load_model(args.model, pretrained=True,
                               checkpoint_path=args.checkpoint, device=args.device,
                               dataset_name=args.dataset)
    baseline_metrics = evaluate(model_tmp, val_batches, task_type, args.device)
    del model_tmp
    torch.cuda.empty_cache()
    print(f"Baseline metrics: {baseline_metrics}")
    
    all_results = {}
    all_final = {}
    for method in methods:
        history, final = run_method(method, args, train_loader, val_batches, task_type)
        all_results[method] = history
        all_final[method] = final
        gc.collect()
        torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    baseline_ppl = baseline_metrics.get('perplexity', None)
    baseline_loss = baseline_metrics['loss']
    for method, final in all_final.items():
        if baseline_ppl and 'perplexity' in final:
            deg = (final['perplexity'] - baseline_ppl) / baseline_ppl * 100
            print(f"  {method}: PPL={final['perplexity']:.2f} (degradation={deg:+.2f}%)")
        else:
            deg = (final['loss'] - baseline_loss) / baseline_loss * 100
            print(f"  {method}: loss={final['loss']:.4f} (degradation={deg:+.2f}%)")
    
    # Save
    flat_results = []
    for method, history in all_results.items():
        for h in history:
            flat_results.append({'method': method, **h})
    
    save_results(flat_results, args.output_dir,
                 f"ft_{args.model}_{args.dataset}_K{args.num_recoveries}",
                 config={**vars(args), 'baseline': baseline_metrics, 'final': all_final})
    plot_results(all_results, args)
    print("\nDone!")


if __name__ == '__main__':
    main()
