# Budget-tangent spectral allocation

All values below were measured on one Tesla V100 32GB. The raw incremental
results are:

- `../spectral_probe_smoke/20260809_142039_gpt2m_recovery_long.json`: one-seed
  100/50-step quality run with eight probe batches and 100 evaluation batches.
- `../spectral_probe_cache_smoke/20260809_143216_gpt2m_recovery_long.json`:
  two-step performance smoke after adding stable score-order caching.
- `../spectral_probe_device_cache_smoke/20260809_143802_gpt2m_recovery_long.json`:
  the same performance smoke after caching current/reference states on-device.
- `../spectral_probe_final_smoke/20260809_144250_gpt2m_recovery_long.json`: the
  final performance smoke with GPU stable sorting.

## Method

Let `p_l` be the pruning rate of Transformer layer `l`, and let `n_l` be its
eligible parameter count. The exact global budget is
`sum_l n_l p_l = B`. Layer sensitivity is treated as a smooth signal along
depth and probed in the first `K` nonconstant DCT modes.

Each DCT mode is projected onto the budget tangent space, normalized, and used
to build paired masks at `p +/- 0.025 d`. The paired loss difference estimates
one directional derivative while preserving the exact budget in both masks.
The first `K` responses reconstruct the minimum-norm gradient component in the
measured subspace. A proximal trust-region step with radius `0.1` then produces
the final exact-budget layer counts. Taylor scores still rank coordinates
within each layer.

This uses `2K` simultaneous whole-model probes instead of the full method's 48
one-layer probes. `K=4` and `K=8` share the first eight measurements; they are
not calibrated in separate runs.

## Quality result

The quality run used seed 42, removed exactly 150,994,944 of 301,989,888
eligible residual parameters, and used the same training and evaluation
batches for every method. Lower perplexity is better.

| Method | Immediate PPL | PPL after 50 matched steps |
|---|---:|---:|
| Residual magnitude + uniform | 24.8261 | 24.2047 |
| First-order + uniform | 24.5241 | 24.0203 |
| Taylor + uniform | 24.5187 | 24.0237 |
| Taylor + Weibull MoM | 24.5265 | 24.0250 |
| Taylor + exact global | 24.5358 | 24.0317 |
| Taylor + full Probe Trust | 24.4854 | 24.0151 |
| Taylor + spectral `K=4` | 24.4863 | **24.0149** |
| Taylor + spectral `K=8` | **24.4838** | 24.0179 |

Immediate `K=8` improved over Taylor + uniform by `0.03491` PPL and over
Weibull by `0.04270` PPL. It also improved over full Probe Trust by `0.00154`
PPL. `K=4` was `0.00092` PPL worse than full Probe immediately.

After 50 matched continuation steps, `K=4` improved over uniform by `0.00875`
PPL, over Weibull by `0.01008` PPL, and over full Probe by `0.00018` PPL.
`K=8` still beat uniform and Weibull by `0.00580` and `0.00713` PPL, but was
`0.00278` PPL worse than full Probe.

The DCT reconstruction was numerically well-conditioned: condition numbers
were `1.033` for `K=4` and `1.152` for `K=8`. Both response residual norms were
below `1.3e-17`.

## Measured overhead

The initial quality implementation exposed a mask-generation bottleneck. Its
spectral calibration took `125.71s`: `107.32s` constructing masks and `17.85s`
applying/evaluating them. It used 16 direction evaluations and 128 batch
forwards, compared with 424 batch forwards for full Probe Trust, but fewer
forwards did not initially translate into lower wall time.

The final implementation preserves exact masks while caching stable Taylor
score orders, sorting them on the GPU, and keeping current/reference eligible
states on the GPU during spectral probing. On the final two-probe-batch smoke:

| Component | Seconds |
|---|---:|
| Shared GPU score-order construction | 2.125 |
| Full Probe Trust calibration | 11.721 |
| Spectral calibration (`K=4,8` together) | 10.136 |
| Spectral device-cache setup | 0.899 |
| Spectral mask construction | 7.283 |
| Spectral mask application and evaluation | 1.830 |
| Complete allocation section, all compared methods | 55.307 |

The spectral core was `13.5%` faster than full Probe Trust in this short
performance protocol. It required a 2,415,919,104-byte CPU order cache and a
2,415,919,104-byte temporary GPU state cache (about 2.25 GiB each). The cached
and final smoke runs produced identical PPL values for every method, and unit
tests verify exact mask equivalence including tied scores.

## Scope

This one-seed result supports the feasibility claim: a low-frequency,
budget-tangent sensitivity model can match the much denser layerwise probe and
can outperform uniform and Weibull allocation on immediate recovery. It does
not establish a multi-seed advantage or a durable `K=8` advantage after
continued training. The performance smoke establishes lower calibration time
under its two-batch protocol, not the paper's current `<1%` end-to-end overhead
claim. The optimized code was not rerun for the full 100/50-step protocol to
avoid another long experiment; its exact-mask equivalence was instead checked
by tests and by identical short-run outputs.
