# Scale-invariant empirical quantile allocation

All results below were measured on one Tesla V100 32GB. The raw incremental
outputs are:

- `../quantile_smooth_schema_smoke/20260809_151822_gpt2m_recovery_long.json`
- `../quantile_smooth_effect_smoke/20260809_152141_gpt2m_recovery_long.json`
- `../quantile_relative_effect_smoke/20260809_152737_gpt2m_recovery_long.json`

## Objective

For layer `l`, let its stable ascending Taylor scores be
`q_l,1 <= ... <= q_l,n`. The empirical cumulative cost is the convex,
piecewise-linear interpolation

```
C_l(p) = sum_{i <= floor(n_l p)} q_l,i
       + (n_l p - floor(n_l p)) q_l,floor(n_l p)+1.
```

The allocator solves

```
min_p sum_l C_l(p_l) / s_l + lambda / 2 * sum_l (p_{l+1} - p_l)^2
```

under the exact global count budget and a +/-10 percentage-point layer-rate
box. The `global` variant uses one shared scale. The scale-invariant
`layer_uniform_cost` variant first computes the exact-budget uniform count
`k_l^uniform` and uses `s_l = L * abs(C_l(k_l^uniform / n_l))`. Thus,
multiplying all scores in a nondegenerate layer by any representable positive
constant does not change the solution. In this 50% experiment the reference
rate is 0.5 up to integer budget rounding. The implementation rejects a layer
whose uniform-point cumulative cost is zero.

The full empirical curves are optimized with proximal gradient. Each prox uses
binary search over the sorted empirical marginal scores and an outer budget
multiplier search. A two-choice chain dynamic program rounds the continuous
rates while preserving the exact integer budget and the smoothness coupling.
No loss probe, validation data, or model forward is used by this allocator.
The first-difference penalty is a DCT/Laplacian low-pass regularizer because its
eigenvalue increases with depth frequency.

## Effect smoke

Both effect runs use seed 42, a 50-step recovery point, four HVP batches, 20
held-out evaluation batches, and one matched continuation step. Their
checkpoint, selected training batches, and evaluation batches are identical.
All methods remove exactly 150,994,944 of 301,989,888 eligible residual
parameters. Lower perplexity is better.

| Method | Immediate PPL | PPL after 1 step |
|---|---:|---:|
| Taylor + uniform | 24.2669 | 24.2666 |
| Taylor + Weibull MoM | 24.2771 | 24.2768 |
| Taylor + exact global | 24.2825 | 24.2823 |
| Taylor + Probe Trust | **24.2395** | **24.2392** |
| Raw empirical, lambda=0.01 | 24.2820 | 24.2817 |
| Raw empirical, lambda=0.1 | 24.2809 | 24.2806 |
| Raw empirical, lambda=1 | 24.2769 | 24.2767 |
| Relative empirical, lambda=0.01 | 24.2650 | 24.2648 |
| Relative empirical, lambda=0.1 | 24.2650 | 24.2647 |
| Relative empirical, lambda=1 | **24.2649** | **24.2646** |

The raw empirical objective reduced the Taylor proxy cost but made real PPL
worse than uniform. Its allocation rates were negatively correlated with the
Probe Trust rates (`r` approximately `-0.50`). This demonstrates that the raw
Taylor score scale is not calibrated across layers; smoothness alone cannot
repair a reversed cross-layer signal.

The raw effect file predates the explicit normalization field and uses method
keys of the form `taylor_quantile_smooth_l*`; these are equivalent to the
current `global` mode. Its `lambda=1` solve stopped after 300 iterations with a
fixed-point residual of `1.58e-9`, just above the configured `1e-9` threshold,
so its metadata correctly reports `converged=false`.

The scale-invariant objective corrected most of this failure. At `lambda=1`,
it improved immediate PPL over uniform by `0.00198` and over Weibull by
`0.01218`. After one matched step, the improvements were `0.00199` and
`0.01219`. It remained `0.02539` PPL worse than Probe Trust immediately.

These are development results, not confirmatory results. The relative
normalization was introduced after diagnosing the raw objective on the same
seed, and the three lambda values were all inspected on the same evaluation
split. No full 100/50-step or held-out-seed run was performed because the
observed gain over uniform is small.

## Cost

For the relative three-lambda effect run:

| Component | Seconds |
|---|---:|
| GPU stable score ordering | 2.128 |
| Sorted score materialization | 2.007 |
| Three convex solves and integer rounding | 2.108 |
| Quantile allocator total | 4.224 |
| Complete allocation section, all compared methods | 49.701 |

The quantile allocator reports zero model and batch-forward evaluations. The
score-order cache uses 2,415,919,104 bytes (about 2.25 GiB) of CPU memory; the
temporarily materialized float32 sorted scores add about 1.125 GiB. A single
fixed lambda would avoid most of the three-solve grid cost, but that timing was
not measured separately.

## Conclusion

The experiment supports two narrow claims. First, a full empirical quantile
curve can replace the Weibull distribution without any loss probe. Second,
cross-layer scale invariance is necessary: the unnormalized empirical objective
fails, while relative normalization slightly beats Weibull and uniform. The
current evidence does not support a large advantage over uniform or a claim
that loss probes are unnecessary for the best observed allocation.
