# Probe-calibrated trust-region allocation

All numbers below were measured on one Tesla V100 32GB. None are synthetic.
The raw, incrementally written results are:

- `../probe_trust_development/20260809_121503_gpt2m_recovery_long.json`
- `../probe_trust_holdout/20260809_122100_gpt2m_recovery_long.json`

## Method

Taylor scores still rank residual coordinates within each Transformer layer.
The Weibull cross-layer allocator is replaced by a local, distribution-free
calibration:

1. Start from the exact-budget uniform mask.
2. On eight training-only probe batches, change one layer at a time by +/-2.5
   percentage points and estimate its finite-difference marginal loss.
3. Take a proximal loss-decreasing allocation step under the exact global
   budget and a symmetric layer-rate trust region.
4. Select the trust radius from `{2.5%, 5%, 10%}` on eight disjoint
   training-only selection batches.
5. Report quality on 100 validation batches that were never used for scoring,
   probing, or radius selection.

Seed 42 was the development seed used to lock the candidate set. Seeds 43 and
44 were then run as held-out seeds without changing the algorithm or candidate
set. Every seed selected the 10% trust radius from its own training-only
selection split. All methods removed exactly 150,994,944 of 301,989,888
eligible residual parameters (50%).

## Results

Lower perplexity is better.

| Method | Immediate PPL | Final PPL |
|---|---:|---:|
| Residual magnitude + uniform | 25.1147 | 24.4098 |
| First-order + uniform | 24.8093 | **24.2324** |
| Taylor + uniform | 24.7998 | 24.2368 |
| Taylor + Weibull MoM | 24.8041 | 24.2393 |
| Taylor + exact global | 24.8165 | 24.2464 |
| Taylor + probe trust region | **24.7681** | 24.2356 |

The probe allocator beat Taylor + uniform immediately in all three seeds by
`0.03338`, `0.03643`, and `0.02530` PPL. Both held-out seeds improved. Across
the combined development and held-out runs, the descriptive mean improvement
was `0.03170` PPL and the exploratory 95% t interval was
`[0.01741, 0.04600]`.

It beat Taylor + Weibull immediately in all three seeds by `0.04116`,
`0.03947`, and `0.02742` PPL. The combined descriptive mean improvement was
`0.03602` PPL with an exploratory 95% t interval of `[0.01741, 0.05463]`.
These are not confirmatory confidence intervals because seed 42 was used to
expand and lock the radius grid; only seeds 43 and 44 were held out.

After 50 matched continuation steps, the paired differences versus uniform
were `-0.00857`, `-0.00009`, and `+0.00520` PPL. The mean improvement was only
`0.00115` PPL and its confidence interval crossed zero. The durable advantage
is therefore not established.

## Diagnosis and scope

The result supports a narrow contribution: training-loss probes can calibrate
cross-layer residual allocation better than a fitted Weibull CDF for immediate
checkpoint restoration. It does not support claiming that the allocator is
globally optimal for actual loss or that its advantage survives continuation
training in every seed.

Calibration took 55.6-55.9 seconds per seed and used 48 layer probes plus three
candidate evaluations. Including the ordinary mask construction, the full
allocation section took 103.3-105.1 seconds. The probe method consumed 16
additional training batches and 424 batch-forward evaluations (108,544 token
presentations), so this is an equal-compression-budget comparison, not an
equal-data or equal-compute comparison. This overhead is material and is
incompatible with the paper's current `<1%` overhead claim without
amortization or a cheaper probe schedule.

All three seeds selected the largest tested radius, so the search has not yet
demonstrated an internal optimum. In addition, seeds 43 and 44 did not exactly
reproduce the older long-run absolute PPL values despite matching checkpoint
and data hashes. Paired comparisons within each run remain controlled, but the
repository must enable deterministic GPU execution before claiming exact
cross-process reproduction.
