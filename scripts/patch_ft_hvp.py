"""Patch run_fault_tolerant_training.py to add proper block-wise HVP support.

This script adds a new code path for 'second-order-hvp' importance that
computes real block-wise Hessian-Vector Products instead of the first-order
approximation |g*w|.

Run on server:
    cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress
    python /tmp/patch_ft_hvp.py
"""
import re

FT_PATH = '/lihongliang/fangzl/ckpt-compress/code/checkpoint_compress/experiments/scripts/run_fault_tolerant_training.py'

with open(FT_PATH, 'r') as f:
    code = f.read()

# 1. Add import for block-wise HVP and cache_batches
old_import = "from dacp.quantization.kmeans import KMeansQuantizer"
new_import = """from dacp.quantization.kmeans import KMeansQuantizer
from dacp.tools.importance import compute_importance_scores_hvp_blockwise"""

if 'compute_importance_scores_hvp_blockwise' not in code:
    code = code.replace(old_import, new_import)
    print("[1] Added import for compute_importance_scores_hvp_blockwise")
else:
    print("[1] Import already present, skipping")

# 2. Replace the second-order-hvp code path to use real HVP
# Find the old ours-2d/second-order-hvp block and add a separate second-order-hvp path before it
old_block = """    # --- ours-2d / second-order-hvp \xe2\x86\x92 2D combo + weibull-adaptive ----
    if imp_name in ('ours-2d', 'second-order-hvp'):
        mag_scorer = get_importance_scorer('magnitude')
        mag_scores = mag_scorer.score(prunable_w, prunable_g)
        damage_scores = {}
        for name in prunable_w:
            w = prunable_w[name]
            g = prunable_g.get(name, torch.zeros_like(w))
            damage_scores[name] = (g * w).abs()
        combined = combine_scores_2d_with_protection(
            mag_scores, damage_scores,
            protection_ratio=args.protection_ratio,
            alpha=args.alpha,
        )
        alloc = alloc_name or 'weibull-adaptive'
        pruner = Pruner(importance='magnitude', allocation=alloc)"""

new_block = """    # --- second-order-hvp: real block-wise HVP scoring ---
    if imp_name == 'second-order-hvp':
        # Build loss function for HVP computation
        _criterion = nn.CrossEntropyLoss()
        if task_type == 'lm':
            def _hvp_loss_fn(m, batch):
                ids = batch['input_ids'].to(device)
                labs = batch['labels'].to(device)
                out = m(ids)
                logits = out.logits if hasattr(out, 'logits') else out
                return _criterion(logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                                  labs[..., 1:].contiguous().view(-1))
        elif task_type == 'cls':
            def _hvp_loss_fn(m, batch):
                ids = batch['input_ids'].to(device)
                labs = batch['labels'].to(device)
                attn = batch.get('attention_mask')
                if attn is not None:
                    attn = attn.to(device)
                out = m(ids, attention_mask=attn)
                logits = out.logits if hasattr(out, 'logits') else out
                return _criterion(logits, labs)
        elif task_type == 'reg':
            _mse = nn.MSELoss()
            def _hvp_loss_fn(m, batch):
                ids = batch['input_ids'].to(device)
                labs = batch['labels'].to(device).float()
                attn = batch.get('attention_mask')
                if attn is not None:
                    attn = attn.to(device)
                out = m(ids, attention_mask=attn)
                logits = out.logits if hasattr(out, 'logits') else out
                return _mse(logits.squeeze(-1), labs)
        else:
            def _hvp_loss_fn(m, batch):
                imgs = batch['images'].to(device)
                labs = batch['labels'].to(device)
                out = m(imgs)
                logits = out.logits if hasattr(out, 'logits') else out
                return _criterion(logits, labs)
        # Collect a few batches for HVP
        hvp_batches = []
        data_iter = iter(train_loader)
        n_hvp = min(getattr(args, 'num_importance_steps', 5), 5)
        for _ in range(n_hvp):
            try:
                b = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                b = next(data_iter)
            hvp_batches.append({k: v.to(device) for k, v in b.items()})
        # Determine model family
        model_name = getattr(args, 'model', 'gpt2')
        if 'bert' in model_name:
            mf = 'bert'
        elif 'pythia' in model_name:
            mf = 'pythia'
        elif 'vit' in model_name:
            mf = 'vit'
        else:
            mf = 'gpt2'
        # Compute block-wise HVP scores
        hvp_scores = compute_importance_scores_hvp_blockwise(
            model, _hvp_loss_fn, hvp_batches,
            model_family=mf, num_batches=min(n_hvp, 3),
            alpha=args.alpha, normalize=False,
        )
        # Filter to prunable params only
        hvp_scores = {k: v for k, v in hvp_scores.items() if k in prunable_w}
        # Use Weibull-adaptive allocation on HVP scores
        alloc = alloc_name or 'weibull-adaptive'
        pruner = Pruner(importance='magnitude', allocation=alloc)
        layer_ratios = pruner.compute_layer_ratios(hvp_scores, args.prune_ratio)
        del weights, grads, prunable_w, prunable_g, hvp_batches
        gc.collect()
        torch.cuda.empty_cache()
        return hvp_scores, layer_ratios

    # --- ours-2d: 2D combo (magnitude + first-order) + weibull-adaptive ---
    if imp_name == 'ours-2d':
        mag_scorer = get_importance_scorer('magnitude')
        mag_scores = mag_scorer.score(prunable_w, prunable_g)
        damage_scores = {}
        for name in prunable_w:
            w = prunable_w[name]
            g = prunable_g.get(name, torch.zeros_like(w))
            damage_scores[name] = (g * w).abs()
        combined = combine_scores_2d_with_protection(
            mag_scores, damage_scores,
            protection_ratio=args.protection_ratio,
            alpha=args.alpha,
        )
        alloc = alloc_name or 'weibull-adaptive'
        pruner = Pruner(importance='magnitude', allocation=alloc)"""

if old_block in code:
    code = code.replace(old_block, new_block)
    print("[2] Patched: added separate second-order-hvp path with real block-wise HVP")
else:
    # Try to find the block with different whitespace
    print("[2] WARNING: Could not find exact old block to replace.")
    print("    Make sure the old 'ours-2d / second-order-hvp' block exists as expected.")

with open(FT_PATH, 'w') as f:
    f.write(code)

print("\nDone. The FT trainer now supports:")
print("  - second-order-hvp+weibull-adaptive: real block-wise HVP scoring")
print("  - ours-2d: original 2D combination (magnitude + first-order)")
print("  - magnitude+weibull-adaptive: magnitude scoring with Weibull allocation")
