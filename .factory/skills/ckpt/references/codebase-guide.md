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

Scripts are organized by function:

### `finetune/` — Training & Fine-tuning
- `train_cv.py` — CV model training (ResNet on CIFAR)
- `train_nlp.py` — NLP model training (GPT-2 on WikiText)
- `finetune_gpt2_1000steps.py` — GPT-2 fine-tuning for checkpoint generation
- `finetune_bert_large_sst2.py` / `finetune_bert_large_mnli.py` / `finetune_bert_large_stsb.py` — BERT fine-tuning

### `compress/` — Checkpoint Compression
- `compress_and_resume.py` — Compress and decompress checkpoints

### `prune/` — Pruning
- `gamma_adaptive_pruning.py` — Gamma-based adaptive pruning (second-order)
- `gamma_adaptive_pruning_first_order.py` — Gamma-based adaptive pruning (first-order)

### `analysis/` — Analysis & Visualization
- `fine_grained_pruning_analysis.py` — GPT-2 fine-grained sub-layer analysis
- `bert_fine_grained_pruning_analysis.py` — BERT fine-grained sub-layer analysis
- `layer_pruning_analysis.py` — Layer-wise pruning rate analysis
- `layer_pruning_analysis_high_sparsity.py` — High sparsity analysis
- `visualize_model_structure.py` — Model structure visualization

### `config/` — Configuration Tools
- `generate_pruning_config.py` — Generate pruning configs
- `generate_global_pruning_config.py` — Generate global pruning configs
- `test_pruning_config.py` — Test pruning configs

### `comparison/` — Comparison Experiments
- `compare_first_vs_second_order.py` — First-order vs combined importance

### Archived Scripts
Exploratory and deprecated scripts are in `archive/scripts/`.

## Configuration: `experiments/configs/`

```
experiments/configs/
├── cv/           — CV training configs (ResNet + CIFAR)
├── nlp/          — NLP training configs (GPT-2 + WikiText)
└── pruning/      — Pruning configs (per-layer rates)
```

## Results Directory

```
results/
├── paper_results/    — Core results for the paper
├── supplementary/    — Supplementary visualizations
└── archive/          — Archived exploratory experiments
```

## Documentation: `docs/`

```
docs/
├── guides/       — Usage guides (EXPERIMENTS, DATA_PREPARATION, BERT_USAGE, etc.)
├── design/       — Design docs (ARCHITECTURE, EXPERIMENT_DESIGN, PLAN)
├── reference/    — Reference docs (MAGNITUDE_IMPORTANCE)
└── archive/      — Archived docs
```
