# Codebase Guide

## Core Module: `src/ckpt_compress/methods/adam_prune/`

### importance.py — Importance Score Computation

Key functions:

- `compute_importance_scores(weights, gradients, exp_avg_sq, alpha=0.5)` — Signed score: `s_i = -g_i·θ_i + α·v_i·θ_i²`
- `compute_importance_scores_abs(weights, gradients, exp_avg_sq, alpha=0.5)` — Absolute value: `|g_i·θ_i| + α·|v_i·θ_i²|`
- `compute_importance_scores_first_order(weights, gradients)` — First-order only: `|g_i·θ_i|`
- `compute_importance_scores_hvp(model, criterion, data_batches, device)` — HVP-based exact Hessian diagonal
- `compute_importance_scores_hvp_memory_efficient(...)` — Memory-efficient HVP for large models

### distribution.py / distribution_fast.py — Distribution Fitting

- Fits Gamma, Log-normal, Exponential, Weibull to per-layer score distributions
- Returns fitted parameters + KS-test goodness-of-fit
- `distribution_fast.py` uses method-of-moments for speed

### adaptive_pruning.py — Gamma-Based Adaptive Pruning

- `compute_adaptive_pruning_rates(scores, global_rate, ...)` — Main entry point
- Fits Gamma per layer, solves bisection for global threshold `c*`
- Returns per-layer pruning rates `p_ℓ = F_ℓ(c*)`

### layer_pruning.py — Layer-Wise Pruning Execution

- `prune_by_layer_rates(state_dict, scores, rates)` — Apply per-layer pruning masks
- `prune_uniform(state_dict, scores, global_rate)` — Uniform pruning baseline
- `prune_global_topk(state_dict, scores, global_rate)` — Global top-k pruning

## Models: `src/ckpt_compress/models/`

### gpt2.py
- `get_gpt2_small()` / `get_gpt2_medium()` — Load GPT-2 with HuggingFace
- Returns model ready for fine-tuning

### bert.py
- `get_bert_base()` / `get_bert_large()` — Load BERT models
- Supports classification head for downstream tasks

### resnet.py
- `get_resnet18()` / `get_resnet50()` — Standard torchvision ResNets for CIFAR

## Data Loaders: `src/ckpt_compress/utils/data_loader.py`

- `get_wikitext2_dataloader(split, batch_size, seq_length)` — WikiText-2
- `get_wikitext103_dataloader(split, batch_size, seq_length)` — WikiText-103
- `get_cifar10_loaders(batch_size)` — CIFAR-10 train/test
- `get_cifar100_loaders(batch_size)` — CIFAR-100 train/test

## Experiment Scripts: `experiments/scripts/`

### Comparison Scripts
- `compare_first_vs_second_order.py` — First-order vs combined importance
- `compare_first_vs_second_order_direct.py` — Direct comparison variant
- `compare_first_vs_second_order_checkpoint.py` — Checkpoint-based comparison
- `compare_three_methods.py` — Three-way comparison (magnitude, first-order, second-order)

### Adaptive Pruning Scripts
- `gamma_adaptive_pruning.py` — Gamma-based adaptive pruning (second-order)
- `gamma_adaptive_pruning_first_order.py` — Gamma-based adaptive pruning (first-order)
- `gamma_adaptive_pruning_multi_ratio.py` — Multi-ratio second-order
- `gamma_adaptive_pruning_first_order_multi_ratio.py` — Multi-ratio first-order

### Analysis Scripts
- `fine_grained_pruning_analysis.py` — GPT-2 fine-grained sub-layer analysis
- `bert_fine_grained_pruning_analysis.py` — BERT fine-grained sub-layer analysis
- `layer_pruning_analysis.py` — Layer-wise pruning rate analysis
- `analyze_importance_distribution.py` — Score distribution visualization
- `analyze_numerical_sensitivity.py` — Numerical sensitivity analysis

### Training Scripts
- `train_cv.py` — CV model training (ResNet on CIFAR)
- `train_nlp.py` — NLP model training (GPT-2 on WikiText)
- `finetune_gpt2_1000steps.py` — GPT-2 fine-tuning for checkpoint generation
- `finetune_bert_large_sst2.py` / `finetune_bert_large_mnli.py` / `finetune_bert_large_stsb.py` — BERT fine-tuning

## Configuration: `experiments/configs/`

YAML configs for reproducible experiments. Check `experiments/configs/` for available presets.

## Results Directory

Results saved to `results/` with timestamp subdirectories. Each experiment typically outputs:
- CSV files with per-layer metrics
- JSON summary files
- PNG figures (loss curves, distribution plots, comparison charts)
