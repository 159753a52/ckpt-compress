# Longer residual recovery experiment

The raw result is `20260809_112042_gpt2m_recovery_long.json`. All values were
measured on one Tesla V100 32GB; none are synthetic.

## Protocol

- GPT-2 Medium / WikiText-103, seeds 42, 43, and 44.
- Start from checkpoint step 1000 and run 100 explicit optimizer steps.
- One recovery after step 50; 50 matched continuation steps after recovery.
- AdamW at `5e-5` with cosine decay over all 100 steps.
- Batch size 2, sequence length 128, and 100 fixed evaluation batches
  (25,600 tokens).
- Eight disjoint score batches; scoring runs in evaluation mode without an
  optimizer update.
- Exactly 150,994,944 of 301,989,888 eligible residual parameters are removed
  by every method (50%).
- Each seed uses disjoint pre-recovery, scoring, and continuation batches.
  Every method within a seed shares the same current model, optimizer state,
  scheduler state, continuation data, evaluation data, and dropout seed.

## Results

Lower perplexity is better. Delta columns are paired differences versus
residual magnitude, so negative values favor the method.

| Method | Immediate PPL | Immediate delta | Final PPL | Final delta |
|---|---:|---:|---:|---:|
| No compression | 25.5310 | - | 24.5441 | - |
| Residual magnitude + uniform | 25.1336 | 0.0000 | 24.3504 | 0.0000 |
| First-order + uniform | 24.8262 | -0.3074 | **24.1630** | **-0.1873** |
| Second-order + uniform | 24.8483 | -0.2853 | 24.1725 | -0.1779 |
| Taylor + uniform | **24.8167** | **-0.3169** | 24.1657 | -0.1847 |
| Taylor + Weibull MoM | 24.8209 | -0.3128 | 24.1684 | -0.1819 |
| Taylor + exact global | 24.8328 | -0.3008 | 24.1756 | -0.1747 |

Taylor + uniform beats residual magnitude for all three seeds. Its paired
immediate delta is -0.3169 PPL (95% CI [-0.3577, -0.2761]); its final delta is
-0.1847 PPL (95% CI [-0.1894, -0.1800]). This is a stable scoring result.

The curvature-specific result is much weaker. Taylor improves over first-order
by only 0.0095 PPL immediately (95% CI [-0.0198, 0.0008]) and is 0.0026 PPL
worse after continuation (95% CI approximately [-0.00001, 0.00527]). All three
final seeds favor first-order. The experiment therefore does not establish a
durable benefit from the HVP term.

Weibull is worse than Taylor + uniform for all three final seeds: mean delta
+0.00279 PPL, 95% CI [+0.00140, +0.00418]. Exact global thresholding minimizes
the additive Taylor proxy, but is also worse in actual final PPL by +0.00998
(95% CI [+0.00389, +0.01606]). Neither result supports the current
distribution-aware allocation quality claim.

## Runtime and checks

- Total wall time: 561.8 seconds (9.4 minutes).
- HVP time: 27.9-28.8 seconds per seed.
- Peak allocated GPU memory during scoring: 3.01 GiB.
- Allocation and six-mask construction: 38.5-42.1 seconds per seed.
- All scoring checksum deltas are exactly zero and no parameter version changes.
- All masks hit the exact same global pruning count.

## Scope

This is a three-seed diagnostic on one model, not a complete paper result. The
pre-recovery segment raises mean PPL from 22.5679 to 25.5310; compression can
therefore help by rolling back harmful updates. Comparisons among masks remain
paired, but absolute improvements over no compression should not be presented
as ordinary restoration robustness. A full validation set, additional models,
and a faithful full baseline are still required for a submission claim.

The evidence supports a narrow statement: gradient/Taylor residual selection
is consistently better than residual magnitude in this protocol. It does not
support claiming that the second-order term or Weibull allocation is the source
of that advantage.
