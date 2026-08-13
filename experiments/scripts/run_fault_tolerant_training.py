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

import argparse
import gc
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dacp.pruning import Pruner, exact_pruning_mask, filter_prunable_params
from dacp.quantization.int4 import INT4Quantizer
from dacp.quantization.kmeans import KMeansQuantizer
from experiments.lib.data import cache_batches, get_data_loaders
from experiments.lib.evaluation import evaluate
from experiments.lib.losses import compute_task_loss
from experiments.lib.models import load_model
from experiments.lib.quantization_runtime import ExperimentQuantizer, quantize_model_parameters
from experiments.lib.residual_runtime import checkpoint_optimizer_state
from experiments.lib.results import save_results

METHODS = [
    "none",
    "random+uniform",
    "magnitude+uniform",
    "magnitude+weibull-adaptive",
    "ours-2d",
]


def train_one_step(model, optimizer, batch, task_type, device):
    """Train one step, return loss."""
    model.train()
    loss = compute_task_loss(model, batch, task_type, device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def collect_gradients_inline(model, optimizer, train_loader, device, num_steps, task_type):
    """Collect gradients without changing model or optimizer state."""
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError(f"num_steps must be a positive integer, got {num_steps}")
    accumulated_grads: dict[str, torch.Tensor] = {}

    data_iter = iter(train_loader)
    model.train()

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        optimizer.zero_grad()

        loss = compute_task_loss(model, batch, task_type, device)

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient = param.grad.detach().cpu()
                if name in accumulated_grads:
                    accumulated_grads[name].add_(gradient)
                else:
                    accumulated_grads[name] = gradient.clone()

    for name in accumulated_grads:
        accumulated_grads[name] /= num_steps

    weights = {name: p.detach().cpu() for name, p in model.named_parameters()}
    return weights, dict(accumulated_grads)


def _is_residual_method(imp_name):
    """Check if the importance method operates on weight residuals."""
    return imp_name in ("excp-residual", "ours-3d-residual", "ours-fo-residual", "ours-so-residual")


def apply_residual_pruning(model, scores, layer_ratios, prev_weights, device):
    """Apply residual pruning: prune delta = w_current - w_prev, reconstruct.

    This implements the ExCP-style residual checkpoint compression:
      1. Compute delta = w_current - w_prev_reconstructed
      2. Keep only the top-(1-ratio) delta elements by score
      3. Reconstruct: w_new = w_prev_reconstructed + delta_pruned
      4. Return updated prev_reconstructed for next recovery cycle
    """
    model_params = dict(model.named_parameters())
    masks = {}
    new_prev = {}

    for name, score in scores.items():
        if name not in model_params or name not in layer_ratios:
            continue
        ratio = layer_ratios[name]
        flat = score.flatten()
        k = int(len(flat) * ratio)
        if k == 0:
            masks[name] = torch.ones_like(score)
            new_prev[name] = model_params[name].data.detach().cpu().clone()
            continue

        mask = exact_pruning_mask(score, k)
        masks[name] = mask

        param = model_params[name]
        w_prev = prev_weights[name].to(device)
        delta = param.data - w_prev
        delta_pruned = delta * mask.to(device)
        param.data.copy_(w_prev + delta_pruned)
        new_prev[name] = param.data.detach().cpu().clone()

    # Params not in scores: keep as-is
    for name, param in model_params.items():
        if name not in new_prev:
            new_prev[name] = param.data.detach().cpu().clone()

    return masks, new_prev


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

        mask = exact_pruning_mask(score, k)
        masks[name] = mask

        param = model_params[name]
        param.data.mul_(mask.to(device))

    return masks


# Skip quantizing embedding and layernorm (only quantize weight matrices)
_SKIP_QUANTIZE = ["embed", "wte", "wpe", "ln_", "LayerNorm", "layernorm"]


def apply_quantize_dequantize(model, masks, quantizer, device):
    """Apply quantize-then-dequantize to surviving (non-pruned) parameters.

    This simulates the lossy compression from quantization by replacing
    the actual values with their quantized-then-dequantized approximation.

    Uses mask-aware quantization: pruned positions (mask=0) are excluded
    from K-means clustering, ensuring all centroids represent meaningful values.
    """
    del device
    quantize_model_parameters(
        model,
        masks,
        quantizer,
        skip_patterns=tuple(_SKIP_QUANTIZE),
    )


def _parse_method(method_name):
    """Parse method string into (importance, allocation, quantizer_suffix).

    Accepted formats:
        magnitude+uniform
        first-order+uniform+kmeans16
        ours-2d+kmeans256
        second-order-hvp+weibull-adaptive
    """
    parts = method_name.split("+")
    quant_suffix = None
    # Detect quantization suffix from tail
    if parts and any(parts[-1].startswith(q) for q in ("kmeans", "int4")):
        quant_suffix = parts.pop()
    if len(parts) == 1:
        return parts[0], None, quant_suffix
    return parts[0], parts[1], quant_suffix


def _segment_lengths(total_steps, num_recoveries):
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError(f"total_steps must be a positive integer, got {total_steps}")
    if (
        isinstance(num_recoveries, bool)
        or not isinstance(num_recoveries, int)
        or num_recoveries < 0
    ):
        raise ValueError(f"num_recoveries must be a non-negative integer, got {num_recoveries}")
    segment_count = num_recoveries + 1
    if total_steps < segment_count:
        raise ValueError("total_steps must provide at least one step per training segment")
    base, remainder = divmod(total_steps, segment_count)
    return [base + (1 if index < remainder else 0) for index in range(segment_count)]


def _history_record(metrics, task_type, step, segment):
    if "loss" not in metrics:
        raise KeyError("Evaluation metrics must contain 'loss'")
    record = {"step": step, "loss": metrics["loss"], "segment": segment}
    metric_key = "perplexity" if task_type == "lm" else "accuracy"
    if metric_key not in metrics:
        raise KeyError(f"{task_type} evaluation metrics must contain {metric_key!r}")
    record[metric_key] = metrics[metric_key]
    return record


def compute_scores_for_method(
    method_name,
    model,
    optimizer,
    train_loader,
    device,
    num_steps,
    task_type,
    args,
    reference_weights=None,
):
    """Compute importance scores based on method name.

    Supports generic ``importance+allocation`` names as well as the
    legacy ``ours-2d`` shorthand (mapped to 2D combination + weibull-adaptive).
    """
    imp_name, alloc_name, _ = _parse_method(method_name)

    weights, grads = collect_gradients_inline(
        model, optimizer, train_loader, device, num_steps, task_type
    )

    prunable_w = filter_prunable_params(weights)
    prunable_g = {k: grads[k] for k in prunable_w if k in grads}

    # --- excp-residual: magnitude of delta (ExCP pruning criterion) ---
    if imp_name == "excp-residual":
        scores = {}
        for name in prunable_w:
            w = prunable_w[name]
            ref = reference_weights.get(name) if reference_weights else None
            if ref is not None:
                delta = w - ref.to(w.device)
                scores[name] = delta.abs()
            else:
                scores[name] = w.abs()
        alloc = alloc_name or "uniform"
        pruner = Pruner(importance="magnitude", allocation=alloc)
        layer_ratios = pruner.compute_layer_ratios(scores, args.prune_ratio)
        _protect = ["classifier", "lm_head", "qa_output", "score"]
        for _n in layer_ratios:
            if any(_p in _n.lower() for _p in _protect):
                layer_ratios[_n] = min(layer_ratios[_n], 0.1)
        del weights, grads, prunable_w, prunable_g
        gc.collect()
        return scores, layer_ratios

    # --- ours-3d-residual: ours-3d scoring on delta (our method in residual mode) ---
    # Variants for component ablation: ours-fo-residual (first-order only),
    # ours-so-residual (second-order only)
    if imp_name in ("ours-3d-residual", "ours-fo-residual", "ours-so-residual"):
        use_fo = imp_name in ("ours-3d-residual", "ours-fo-residual")
        use_so = imp_name in ("ours-3d-residual", "ours-so-residual")
        from dacp.tools.importance import (
            build_transformer_blocks,
            complete_blockwise_vector,
            compute_hvp_blockwise_batched,
            include_parameter_blocks,
        )

        def loss_fn(m, batch):
            return compute_task_loss(m, batch, task_type, device)

        score_batches = []
        data_it = iter(train_loader)
        for _ in range(min(8, max(num_steps, 1))):
            try:
                score_batches.append(next(data_it))
            except StopIteration:
                data_it = iter(train_loader)
                score_batches.append(next(data_it))
        _ml = args.model.lower()
        if "bert" in _ml:
            model_family = "bert"
        elif "vit" in _ml:
            model_family = "vit"
        elif "pythia" in _ml:
            model_family = "pythia"
        else:
            model_family = "gpt2"

        # Residual (delta) w.r.t. prev reconstructed reference, on device.
        # The pruning perturbation only touches prunable params, so the
        # HVP probe vector is delta on prunable coords and zero elsewhere.
        model_params = dict(model.named_parameters())
        delta_dev = {}
        for name in prunable_w:
            p = model_params[name]
            ref = reference_weights.get(name) if reference_weights else None
            if ref is not None:
                delta_dev[name] = (p.data - ref.to(p.device)).detach()
            else:
                delta_dev[name] = p.data.detach().clone()

        # Clean gradients at the FINAL weights (eval mode, no optimizer step),
        # replacing the path-averaged stale grads from collect_gradients_inline.
        was_training = model.training
        model.eval()
        clean_grads = {}
        if use_fo:
            grad_param_names = list(delta_dev.keys())
            grad_params = [model_params[n] for n in grad_param_names]
            clean_grads = {n: torch.zeros_like(delta_dev[n]) for n in grad_param_names}
            for b in score_batches:
                loss = loss_fn(model, b)
                gs = torch.autograd.grad(loss, grad_params, allow_unused=True)
                for n, g in zip(grad_param_names, gs):
                    if g is not None:
                        clean_grads[n] += g.detach()
                del loss, gs
            for n in clean_grads:
                clean_grads[n] /= len(score_batches)

        if use_so:
            blocks = include_parameter_blocks(
                build_transformer_blocks(model, model_family),
                prunable_w,
            )
            probe = complete_blockwise_vector(model, blocks, delta_dev)
            hvp_result = compute_hvp_blockwise_batched(
                model,
                loss_fn,
                score_batches,
                blocks,
                num_batches=min(5, len(score_batches)),
                vector=probe,
            )
        else:
            hvp_result = {}
        if was_training:
            model.train()

        # Faithful Taylor damage of reverting delta_i:
        #   ΔL ≈ -g_i·δ_i + 0.5·δ_i·(H·δ)_i ; score = |signed sum|
        hvp_hits = 0
        scores = {}
        for name in prunable_w:
            delta = delta_dev[name]
            g = clean_grads.get(name)
            if use_fo and g is None:
                raise RuntimeError(f"Gradient coverage is missing prunable parameter {name!r}")
            fo = -(g * delta) if g is not None else torch.zeros_like(delta)
            h = hvp_result.get(name)
            if use_so and h is None:
                raise RuntimeError(f"HVP coverage is missing prunable parameter {name!r}")
            if h is not None:
                so = 0.5 * (delta * h.to(delta.device))
                hvp_hits += 1
            else:
                so = torch.zeros_like(delta)
            scores[name] = (fo + so).abs().cpu()
        if use_so:
            print(f"  [{imp_name}] HVP coverage: {hvp_hits}/{len(prunable_w)} params")
        alloc = alloc_name or "weibull-adaptive"
        pruner = Pruner(importance="magnitude", allocation=alloc)
        layer_ratios = pruner.compute_layer_ratios(scores, args.prune_ratio)
        _protect = ["classifier", "lm_head", "qa_output", "score"]
        for _n in layer_ratios:
            if any(_p in _n.lower() for _p in _protect):
                layer_ratios[_n] = min(layer_ratios[_n], 0.1)
        del weights, grads, prunable_w, prunable_g, score_batches, hvp_result
        del delta_dev, clean_grads
        gc.collect()
        torch.cuda.empty_cache()
        return scores, layer_ratios

    # --- ours-2d: magnitude-based scoring + first-order correction + Weibull allocation ---
    if imp_name == "ours-2d":
        missing_gradients = sorted(set(prunable_w).difference(prunable_g))
        if missing_gradients:
            raise RuntimeError(
                f"Gradient coverage is missing prunable parameters: {missing_gradients}"
            )
        scores = {}
        for name in prunable_w:
            w = prunable_w[name]
            g = prunable_g[name]
            mag = w.abs()
            # First-order sensitivity as multiplicative correction
            fo = (g * w).abs()
            fo_mean = fo.mean()
            if fo_mean > 0:
                correction = 1.0 + 0.3 * (fo / (fo_mean + 1e-12))
            else:
                correction = 1.0
            scores[name] = mag * correction
        alloc = alloc_name or "weibull-adaptive"
        pruner = Pruner(importance="magnitude", allocation=alloc)
        layer_ratios = pruner.compute_layer_ratios(scores, args.prune_ratio)
        # Protect output/classifier layers from aggressive pruning
        _protect = ["classifier", "lm_head", "qa_output", "score"]
        for _n in layer_ratios:
            if any(_p in _n.lower() for _p in _protect):
                layer_ratios[_n] = min(layer_ratios[_n], 0.1)
        del weights, grads, prunable_w, prunable_g
        gc.collect()
        return scores, layer_ratios

    # --- ours-3d: magnitude + first-order + second-order HVP correction + Weibull allocation ---
    if imp_name == "ours-3d":
        from dacp.tools.importance import (
            build_transformer_blocks,
            complete_blockwise_vector,
            compute_hvp_blockwise_batched,
            include_parameter_blocks,
        )

        # Phase 1: Build HVP data batches
        def loss_fn(m, batch):
            return compute_task_loss(m, batch, task_type, device)

        hvp_batches = []
        data_it = iter(train_loader)
        for _ in range(min(5, num_steps)):
            try:
                hvp_batches.append(next(data_it))
            except StopIteration:
                data_it = iter(train_loader)
                hvp_batches.append(next(data_it))

        # Phase 2: Block-wise HVP computation
        _ml = args.model.lower()
        if "bert" in _ml:
            model_family = "bert"
        elif "vit" in _ml:
            model_family = "vit"
        elif "pythia" in _ml:
            model_family = "pythia"
        else:
            model_family = "gpt2"
        missing_gradients = sorted(set(prunable_w).difference(prunable_g))
        if missing_gradients:
            raise RuntimeError(
                f"Gradient coverage is missing prunable parameters: {missing_gradients}"
            )
        blocks = include_parameter_blocks(
            build_transformer_blocks(model, model_family),
            prunable_w,
        )
        probe = complete_blockwise_vector(
            model,
            blocks,
            {name: dict(model.named_parameters())[name].detach() for name in prunable_w},
        )
        hvp_result = compute_hvp_blockwise_batched(
            model,
            loss_fn,
            hvp_batches,
            blocks,
            num_batches=len(hvp_batches),
            vector=probe,
        )

        # Phase 3: Combine magnitude + first-order + second-order
        alpha1 = 0.3  # first-order correction weight
        alpha2 = 0.2  # second-order correction weight
        scores = {}
        for name in prunable_w:
            w = prunable_w[name]
            g = prunable_g[name]
            mag = w.abs()
            # First-order correction: |g * w|
            fo = (g * w).abs()
            fo_mean = fo.mean()
            fo_corr = alpha1 * (fo / (fo_mean + 1e-12)) if fo_mean > 0 else 0.0
            # Second-order correction: |theta * HVP|
            if name not in hvp_result:
                raise RuntimeError(f"HVP coverage is missing prunable parameter {name!r}")
            hvp = hvp_result[name].to(w.device)
            so = (w * hvp).abs()
            so_mean = so.mean()
            so_corr = alpha2 * (so / (so_mean + 1e-12)) if so_mean > 0 else 0.0
            scores[name] = mag * (1.0 + fo_corr + so_corr)

        alloc = alloc_name or "weibull-adaptive"
        pruner = Pruner(importance="magnitude", allocation=alloc)
        layer_ratios = pruner.compute_layer_ratios(scores, args.prune_ratio)
        # Protect output/classifier layers from aggressive pruning
        _protect = ["classifier", "lm_head", "qa_output", "score"]
        for _n in layer_ratios:
            if any(_p in _n.lower() for _p in _protect):
                layer_ratios[_n] = min(layer_ratios[_n], 0.1)
        del weights, grads, prunable_w, prunable_g, hvp_batches, hvp_result
        gc.collect()
        return scores, layer_ratios

    # --- second-order-hvp: real block-wise HVP for damage scoring ---
    if imp_name == "second-order-hvp":
        from dacp.tools.importance import compute_importance_scores_hvp_blockwise

        def loss_fn(m, batch):
            return compute_task_loss(m, batch, task_type, device)

        hvp_batches = []
        data_it = iter(train_loader)
        for _ in range(min(5, num_steps)):
            try:
                hvp_batches.append(next(data_it))
            except StopIteration:
                data_it = iter(train_loader)
                hvp_batches.append(next(data_it))
        _ml2 = args.model.lower()
        if "bert" in _ml2:
            model_family = "bert"
        elif "vit" in _ml2:
            model_family = "vit"
        elif "pythia" in _ml2:
            model_family = "pythia"
        else:
            model_family = "gpt2"
        scores = compute_importance_scores_hvp_blockwise(
            model,
            loss_fn,
            hvp_batches,
            model_family=model_family,
            num_batches=len(hvp_batches),
            alpha=0.5,
            normalize=False,
        )
        scores = filter_prunable_params(scores)
        alloc = alloc_name or "weibull-adaptive"
        pruner = Pruner(importance="magnitude", allocation=alloc)
        layer_ratios = pruner.compute_layer_ratios(scores, args.prune_ratio)
        del weights, grads, prunable_w, prunable_g, hvp_batches
        gc.collect()
        return scores, layer_ratios

    # --- Generic path: magnitude / first-order / residual-magnitude ---
    if imp_name == "first-order":
        missing_gradients = sorted(set(prunable_w).difference(prunable_g))
        if missing_gradients:
            raise RuntimeError(
                f"Gradient coverage is missing prunable parameters: {missing_gradients}"
            )
    alloc = alloc_name or "uniform"
    pruner = Pruner(importance=imp_name, allocation=alloc)
    scores = pruner.compute_scores(prunable_w, prunable_g, reference_weights=reference_weights)
    layer_ratios = pruner.compute_layer_ratios(scores, args.prune_ratio)

    del weights, grads, prunable_w, prunable_g
    gc.collect()
    return scores, layer_ratios


def run_method(method_name, args, train_loader, val_batches, task_type):
    """Run a complete training process for one method."""
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")

    imp_name, _, quant_suffix = _parse_method(method_name)

    use_pretrained = not getattr(args, "random_init", False)
    model, _ = load_model(
        args.model,
        pretrained=use_pretrained,
        checkpoint_path=args.checkpoint if use_pretrained else None,
        device=args.device,
        dataset_name=args.dataset,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Load optimizer state from checkpoint if available
    if args.checkpoint is not None:
        optimizer.load_state_dict(checkpoint_optimizer_state(Path(args.checkpoint)))
        for _state in optimizer.state.values():
            for _k, _v in _state.items():
                if isinstance(_v, torch.Tensor):
                    _state[_k] = _v.to(args.device)
        # Reset lr to configured value (checkpoint may have decayed lr to ~0)
        for _pg in optimizer.param_groups:
            _pg["lr"] = args.lr
        print(f"  Loaded optimizer state from checkpoint (lr reset to {args.lr})")

    # Learning rate scheduler
    scheduler = None
    if getattr(args, "lr_schedule", "constant") == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=0)

    # Setup quantizer if needed (via parsed suffix)
    quantizer: ExperimentQuantizer | None = None
    if quant_suffix and quant_suffix.startswith("kmeans"):
        n_clusters = int(quant_suffix.replace("kmeans", "") or "16")
        quantizer = KMeansQuantizer(n_clusters=n_clusters)
    elif quant_suffix == "int4":
        quantizer = INT4Quantizer()

    # Save initial weight snapshot for residual methods
    reference_weights = None
    if imp_name == "residual-magnitude" or _is_residual_method(imp_name):
        reference_weights = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        if _is_residual_method(imp_name):
            print(f"  [Residual mode] Initialized prev_reconstructed snapshot")

    segment_lengths = _segment_lengths(args.total_steps, args.num_recoveries)
    loss_history = []
    global_step = 0
    data_iter = iter(train_loader)

    for seg, segment_steps in enumerate(segment_lengths):
        print(f"\n--- Segment {seg+1}/{args.num_recoveries+1} ---")

        for step in range(segment_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            loss = train_one_step(model, optimizer, batch, task_type, args.device)
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            if step % args.eval_interval == 0:
                metrics = evaluate(model, val_batches, task_type, args.device)
                loss_history.append(_history_record(metrics, task_type, global_step, seg))

        # Compress at recovery point (skip last segment)
        if seg < args.num_recoveries and method_name != "none":
            print(f"  [Recovery {seg+1}] Compressing checkpoint...")

            # Compute scores
            scores, layer_ratios = compute_scores_for_method(
                method_name,
                model,
                optimizer,
                train_loader,
                args.device,
                args.num_importance_steps,
                task_type,
                args,
                reference_weights=reference_weights,
            )

            # Apply pruning
            if _is_residual_method(imp_name):
                masks, reference_weights = apply_residual_pruning(
                    model, scores, layer_ratios, reference_weights, args.device
                )
                print(f"  [Residual mode] Updated prev_reconstructed")
            else:
                masks = apply_pruning_to_model(model, scores, layer_ratios, args.device)

            # Apply quantization if configured
            if quantizer is not None:
                apply_quantize_dequantize(model, masks, quantizer, args.device)

            # Update reference_weights for residual-magnitude recovery
            if imp_name == "residual-magnitude":
                reference_weights = {
                    n: p.detach().cpu().clone() for n, p in model.named_parameters()
                }

            # Release scoring intermediates
            del scores, layer_ratios
            gc.collect()

            metrics = evaluate(model, val_batches, task_type, args.device)
            if "perplexity" in metrics:
                ppl_str = f"PPL={metrics['perplexity']:.2f}"
            elif "accuracy" in metrics:
                ppl_str = f"Acc={metrics['accuracy']:.4f}, loss={metrics['loss']:.4f}"
            else:
                ppl_str = f"loss={metrics['loss']:.4f}"
            print(f"  After compression: {ppl_str}")

        # Re-align RNG at each recovery boundary (all methods, incl. none):
        # scoring paths consume different amounts of CPU/CUDA RNG, which would
        # otherwise fork post-compression trajectories across methods.
        if seg < args.num_recoveries and getattr(args, "seed", None) is not None:
            _s = args.seed + 9973 * (seg + 1)
            torch.manual_seed(_s)
            torch.cuda.manual_seed_all(_s)
            random.seed(_s)
            np.random.seed(_s)

    # Final evaluation
    final_metrics = evaluate(model, val_batches, task_type, args.device)
    print(f"\nFinal: {final_metrics}")

    return loss_history, final_metrics


def plot_results(all_results, args):
    """Plot loss curves."""
    plt.figure(figsize=(12, 6))
    colors = {
        "none": "black",
        "random+uniform": "lightgray",
        "magnitude+uniform": "blue",
        "magnitude+weibull-adaptive": "green",
        "ours-2d": "red",
        "ours-2d+kmeans16": "darkred",
        "first-order+uniform": "gray",
        "first-order+weibull-adaptive": "orange",
    }

    for method, history in all_results.items():
        steps = [h["step"] for h in history]
        metric_key = (
            "perplexity" if "perplexity" in history[0] and history[0]["perplexity"] > 0 else "loss"
        )
        values = [h[metric_key] for h in history]
        plt.plot(steps, values, label=method, color=colors.get(method, "purple"), linewidth=2)

    segment_lengths = _segment_lengths(args.total_steps, args.num_recoveries)
    boundary = 0
    for segment_steps in segment_lengths[:-1]:
        boundary += segment_steps
        plt.axvline(x=boundary, color="gray", linestyle="--", alpha=0.5)

    plt.xlabel("Training Steps")
    plt.ylabel("Validation PPL / Loss")
    plt.title(
        f"Fault-Tolerant Training: {args.model} on {args.dataset}\n"
        f"(K={args.num_recoveries} recoveries, prune={args.prune_ratio})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"ft_{args.model}_{args.dataset}_K{args.num_recoveries}.png", dpi=300)
    plt.savefig(out_dir / f"ft_{args.model}_{args.dataset}_K{args.num_recoveries}.pdf")
    plt.close()
    print(f"Figures saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fault-tolerant training simulation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--num_recoveries", type=int, default=10)
    parser.add_argument("--prune_ratio", type=float, default=0.3)
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method list")
    parser.add_argument("--num_importance_steps", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--alpha", type=float, default=0.7, help="2D scoring weight for magnitude (0-1)"
    )
    parser.add_argument("--protection_ratio", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/paper_results/fault_tolerant")
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="Use randomly initialized model instead of pretrained",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="constant",
        choices=["constant", "cosine"],
        help="Learning rate schedule",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")] if args.methods else METHODS

    print("=" * 60)
    print("Fault-Tolerant Training Simulation")
    print(f"Model: {args.model} | Dataset: {args.dataset}")
    print(f"Steps: {args.total_steps} | Recoveries: {args.num_recoveries}")
    print(f"Prune ratio: {args.prune_ratio} | Methods: {methods}")
    print(f"Alpha: {args.alpha} | Protection: {args.protection_ratio}")
    print("=" * 60)

    train_loader, val_loader, task_type = get_data_loaders(
        args.model,
        args.dataset,
        args.batch_size,
        args.seq_length,
        data_dir=args.data_dir or "./data",
    )

    # Use more val batches for classification to improve accuracy resolution
    n_val = 250 if task_type == "cls" else 50
    val_batches = cache_batches(val_loader, n_val, task_type)

    # Baseline evaluation
    use_pretrained = not args.random_init
    model_tmp, _ = load_model(
        args.model,
        pretrained=use_pretrained,
        checkpoint_path=args.checkpoint if use_pretrained else None,
        device=args.device,
        dataset_name=args.dataset,
    )
    baseline_metrics = evaluate(model_tmp, val_batches, task_type, args.device)
    del model_tmp
    torch.cuda.empty_cache()
    print(f"Baseline metrics: {baseline_metrics}")

    all_results = {}
    all_final = {}
    for method in methods:
        if args.seed is not None:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
        history, final = run_method(method, args, train_loader, val_batches, task_type)
        all_results[method] = history
        all_final[method] = final
        gc.collect()
        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    baseline_ppl = baseline_metrics.get("perplexity", None)
    baseline_loss = baseline_metrics["loss"]
    for method, final in all_final.items():
        if baseline_ppl and "perplexity" in final:
            deg = (final["perplexity"] - baseline_ppl) / baseline_ppl * 100
            print(f"  {method}: PPL={final['perplexity']:.2f} (degradation={deg:+.2f}%)")
        else:
            deg = (final["loss"] - baseline_loss) / baseline_loss * 100
            print(f"  {method}: loss={final['loss']:.4f} (degradation={deg:+.2f}%)")

    # Save
    flat_results = []
    for method, history in all_results.items():
        for h in history:
            flat_results.append({"method": method, **h})

    save_results(
        flat_results,
        args.output_dir,
        f"ft_{args.model}_{args.dataset}_K{args.num_recoveries}",
        config={**vars(args), "baseline": baseline_metrics, "final": all_final},
    )
    plot_results(all_results, args)
    print("\nDone!")


if __name__ == "__main__":
    main()
