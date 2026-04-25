"""Patch: Replace lines 247-269 of run_fault_tolerant_training.py
with proper HVP support."""
import sys

FT = '/lihongliang/fangzl/ckpt-compress/code/checkpoint_compress/experiments/scripts/run_fault_tolerant_training.py'

with open(FT, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Verify expected content (1-indexed line 246 = 0-indexed index 245)
assert 'ours-2d / second-order-hvp' in lines[245] or 'ours-2d' in lines[246], \
    f"Expected ours-2d comment near line 246, got: {lines[245]}"

# Find the start index (the comment line)
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if 'ours-2d / second-order-hvp' in line and '# ---' in line:
        start_idx = i
    if start_idx is not None and i > start_idx and 'alloc = alloc_name or' in line and "'uniform'" in line:
        end_idx = i
        break

assert start_idx is not None, "Could not find start of ours-2d block"
assert end_idx is not None, f"Could not find end of ours-2d block (start={start_idx})"
print(f"  Replacing lines {start_idx+1}-{end_idx} (1-indexed)")  

new_block = '''\
    # --- second-order-hvp: real block-wise HVP scoring ---
    if imp_name == 'second-order-hvp':
        import torch.nn as _nn
        _criterion = _nn.CrossEntropyLoss()
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
            _mse = _nn.MSELoss()
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
        hvp_batches_list = []
        data_iter = iter(train_loader)
        n_hvp = min(getattr(args, 'num_importance_steps', 5), 5)
        for _ in range(n_hvp):
            try:
                b = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                b = next(data_iter)
            hvp_batches_list.append({k: v.to(device) for k, v in b.items()})
        model_name = getattr(args, 'model', 'gpt2')
        if 'bert' in model_name:
            mf = 'bert'
        elif 'pythia' in model_name:
            mf = 'pythia'
        elif 'vit' in model_name:
            mf = 'vit'
        else:
            mf = 'gpt2'
        hvp_scores = compute_importance_scores_hvp_blockwise(
            model, _hvp_loss_fn, hvp_batches_list,
            model_family=mf, num_batches=min(n_hvp, 3),
            alpha=args.alpha, normalize=False,
        )
        hvp_scores = {k: v for k, v in hvp_scores.items() if k in prunable_w}
        alloc = alloc_name or 'weibull-adaptive'
        pruner = Pruner(importance='magnitude', allocation=alloc)
        layer_ratios = pruner.compute_layer_ratios(hvp_scores, args.prune_ratio)
        del weights, grads, prunable_w, prunable_g, hvp_batches_list
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
        pruner = Pruner(importance='magnitude', allocation=alloc)
        layer_ratios = pruner.compute_layer_ratios(damage_scores, args.prune_ratio)
        del weights, grads, prunable_w, prunable_g, mag_scores, damage_scores
        gc.collect()
        return combined, layer_ratios

'''

# Also add import if not present
import_line = "from dacp.tools.importance import compute_importance_scores_hvp_blockwise\n"
for i, line in enumerate(lines):
    if 'compute_importance_scores_hvp_blockwise' in line:
        break
else:
    # Add after the last dacp import
    for i in range(len(lines)-1, -1, -1):
        if 'from dacp.' in lines[i]:
            lines.insert(i+1, import_line)
            print(f"  Added HVP import after line {i+1}")
            # Adjust indices since we inserted a line
            start_idx += 1
            end_idx += 1
            break

new_lines = lines[:start_idx] + [new_block] + lines[end_idx:]

with open(FT, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Patch applied successfully!")
print(f"  Old: lines 247-269 (combined ours-2d/second-order-hvp block)")
print(f"  New: separate second-order-hvp (real HVP) and ours-2d (first-order combo)")
