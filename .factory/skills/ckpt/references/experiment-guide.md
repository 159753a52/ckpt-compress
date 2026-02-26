# Experiment Guide

## Experiment 1: Second-Order vs First-Order Importance

### Goal
Demonstrate that damage scores incorporating Hessian diagonal (`|−g_i·θ_i + 0.5·h_i·θ_i²|`) yield lower loss increase after pruning than first-order only (`|g_i·θ_i|`).

### Protocol

1. Load a fine-tuned checkpoint (with optimizer state)
2. Cache N evaluation batches (ensure identical data for both methods)
3. Compute gradients by accumulating over M training steps
4. Compute Hessian diagonal via HVP or Adam `exp_avg_sq`
5. Compute importance scores: first-order and second-order
6. For each sparsity ratio in {10%, 20%, 30%, 40%, 50%}:
   a. Prune using first-order scores → evaluate loss/accuracy
   b. Prune using second-order scores → evaluate loss/accuracy
7. Report Δloss = loss_pruned − loss_original for each method and ratio

### Key Variables to Tune

| Variable | Range | Notes |
|----------|-------|-------|
| alpha (second-order weight) | 0.1–2.0 | Default 0.5; tune if second-order doesn't win |
| num_steps (gradient accumulation) | 50–200 | More steps = less noisy gradients |
| eval_batches | 5–20 | More batches = more stable loss estimate |
| sparsity ratios | 10%–70% | Second-order advantage grows with sparsity |
| Hessian method | Adam proxy / HVP | HVP is more accurate |

### Expected Outcome
At moderate-to-high sparsity (≥20%), second-order pruning should show smaller loss increase than first-order. The gap should widen as sparsity increases.

### Example Command
```bash
python experiments/scripts/compare_first_vs_second_order.py \
    --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
    --num_steps 100 \
    --prune_ratios 0.10,0.20,0.30,0.40,0.50 \
    --eval_batches 10 \
    --batch_size 4 \
    --seq_length 512 \
    --device cuda
```

### If Results Are Not Clear

1. Increase gradient accumulation steps to 200
2. Switch from Adam proxy to HVP for Hessian
3. Try alpha values: 0.1, 0.3, 0.5, 1.0, 2.0
4. Use a mid-training checkpoint (not fully converged)
5. Increase eval batches to 20 for more stable measurement
6. Try GPT-2 Medium or BERT-Large (larger models may show clearer differences)
7. Ensure bias parameters are excluded from pruning

---

## Experiment 2: Distributional Relaxation vs Uniform Pruning

### Goal
Demonstrate that Gamma-based adaptive per-layer pruning rate allocation yields lower total damage than uniform pruning at the same global sparsity.

### Protocol

1. Load checkpoint, compute importance scores (use second-order)
2. For each global sparsity ratio in {10%, 20%, 30%, 40%, 50%}:
   a. **Uniform**: Apply same pruning rate to all layers → evaluate
   b. **Gamma-adaptive**: Fit Gamma per layer, solve bisection for c*, compute per-layer rates → evaluate
   c. **Global top-k** (optional): Sort all scores globally, prune lowest → evaluate (oracle baseline)
3. Report loss/accuracy for each method and ratio
4. Report per-layer pruning rates from Gamma-adaptive (show heterogeneity)

### Key Variables to Tune

| Variable | Range | Notes |
|----------|-------|-------|
| Distribution family | Gamma / Log-normal / Weibull | Gamma is default; check KS-test fit |
| Granularity | per-layer / per-sub-layer | Sub-layer (Q,K,V,O,FFN1,FFN2) captures more heterogeneity |
| Global sparsity | 10%–50% | Advantage most visible at 20%–40% |
| Bisection tolerance | 1e-6 to 1e-8 | Tighter = more precise threshold |

### Expected Outcome
Gamma-adaptive should match or beat uniform at all sparsity levels, with the gap increasing at higher sparsity. It should approach global top-k performance (the oracle).

### Example Command
```bash
python experiments/scripts/gamma_adaptive_pruning_multi_ratio.py \
    --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
    --num_steps 100 \
    --global_prune_ratios 0.10,0.20,0.30,0.40,0.50 \
    --eval_batches 10 \
    --batch_size 4 \
    --device cuda
```

### If Results Are Not Clear

1. Use sub-layer granularity instead of per-layer
2. Verify Gamma fit quality (KS p-value > 0.05 for most layers)
3. Try a model with more architectural diversity (BERT-Large > GPT-2 Small)
4. Increase sparsity to 40%+ where allocation matters more
5. Add magnitude-based uniform as a weaker baseline
6. Visualize per-layer rate allocations to confirm they are non-uniform

---

## Experiment 3: Combined (Second-Order + Gamma-Adaptive)

### Goal
Show the full method (second-order scores + Gamma-adaptive allocation) outperforms all baselines.

### Baselines
1. Magnitude-based uniform pruning
2. First-order uniform pruning
3. First-order Gamma-adaptive pruning
4. Second-order uniform pruning

### Protocol
Run all 5 methods (4 baselines + full method) on the same checkpoint with identical data. Report results in a single table.

---

## Workloads

### NLP
- **GPT-2 Small/Medium on WikiText-103**: Metric = perplexity (lower is better)
- **BERT-Large on SST-2**: Metric = accuracy (higher is better)
- **BERT-Large on MNLI**: Metric = accuracy
- **BERT-Large on STS-B**: Metric = Pearson/Spearman correlation

### CV
- **ResNet-18 on CIFAR-10**: Metric = top-1 accuracy
- **ResNet-50 on CIFAR-100**: Metric = top-1 accuracy

---

## Figure Generation

Key figures for the paper:

1. **Fig 1**: Actual vs predicted Δloss (already exists as `figs/fig1_actual_vs_predicted.png`)
2. **Comparison table**: Loss/accuracy at multiple sparsity levels for all methods
3. **Per-layer rate visualization**: Bar chart showing Gamma-adaptive rates vs uniform
4. **Score distribution plots**: Histograms per layer with Gamma fit overlay
5. **Sparsity-accuracy curve**: Line plot comparing methods across sparsity levels
