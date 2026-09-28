# Coverage-tracked probe scoring on a 32B model: fixed protocol

Status: fixed before the confirmatory runs. Any later change is appended to
the "Deviations" section with its reason; results are never filtered.

## Question

Can the exact block-wise HVP in DACP's residual score be replaced by a small
number of model-distribution probe backwards, and does the probability mass
covered by the sampled tokens tell us when to stop?  The first-order term
`-g ⊙ d` with the true-label gradient is kept unchanged in every Taylor scorer.

Curvature objects that must not be conflated:

| Name | Definition | Used by |
|---|---|---|
| Exact Hessian | `∇²L` restricted to one decoder layer (block diagonal) | `hvp_global`, `dacp` |
| Gauss–Newton / model Fisher | `F = J^T C_o J`, `C_o = blockdiag(diag p_t − p_t p_tᵀ)/N` | probe scorer (full, cross-layer) |
| Empirical Fisher | outer products of true-label gradients | not used |

The probe scorer differs from the HVP scorer in two ways (Hessian → GN and
block-diagonal → full), so the comparison measures their joint effect.

## Probe estimator

Round k samples `y_t ~ p_t` independently at every scoring position
(with replacement across rounds).  `S_t` is the de-duplicated set of sampled
tokens and `q_t = Σ_{y∈S_t} p_t(y)`.  Only a new token contributes, with
backward weight `c_t = ξ_t sqrt(p_t(y)) / sqrt(N)`, `ξ_t` Rademacher.  One
ordinary backward of `Σ_t c_t CE_t` gives `w_k`, and `u = Σ_k w_k (w_k·d)`.
Conditional on the sets, `E_ξ[Σ_k w_k w_kᵀ] = F_S`, the probability-weighted
truncation of `F` to the sampled tokens (unit tests check this identity on an
explicit model).  Score: `s = |−g⊙d + ½ d⊙u|`.

Truncation bound (conditions: every `||J_tᵀ(e_y − p_t)|| ≤ G`):
`0 ≼ F − F_S` and `||F − F_S||₂ ≤ (1 − mean_t q_t) G²`.  It does not bound the
Rademacher estimation error, the Hessian/GN gap, the mask, or the loss.

Stopping rules, evaluated from the same nested probe sequence (max 8 rounds;
if unmet, round 8 is used and the rule is logged as unmet):

- `mean90`, `mean95`: first round with `mean_t q_t ≥ 0.90 / 0.95`
- `p05_90`: first round whose 5th-percentile `q_t ≥ 0.90`
- `stable99`: first round `m ≥ 2` whose 95%-sparsity global mask overlaps the
  round `m−1` mask by at least 99%

Fixed rounds `m = 1..5` are evaluated in every seed.

## Model, data, checkpoints

- Model: DeepSeek-R1-Distill-Qwen-32B (bf16), index SHA-256 recorded per run.
- Data: WikiText-2 raw text files on the host (SHA-256 recorded), tokenized
  once with the model tokenizer and cut into 1024-token chunks.
- Training (tracked code): embedding and LM head frozen; decoder layers and
  final norm trained with AdamW (lr 1e-5, betas 0.9/0.999, eps 1e-8, weight
  decay 0.01 on matrices, clip 1.0, batch 1 × 1024).  Parameters and moments
  are bf16; updates are computed in fp32 and written back with stochastic
  rounding (seeded by step and parameter index).  Round-to-nearest bf16 was
  rejected because an earlier run left 96.6% of residual entries exactly zero.
- Reference `b = θ_100`, checkpoint `θ_200`; both saved losslessly with moments
  and SHA-256 hashes.  Residual `d = θ_200 − b` on eligible matrices
  (q/k/v/o/gate/up/down of all 64 layers); biases and norms are stored
  losslessly and are not counted in sparsity.
- Chunks: training 0–199, recovery 200–349, scoring 1000–1003, gate 1100–1107,
  validation = first 32 validation chunks.

## Sparsity, storage, allocation

- Residual sparsity `P` = fraction of eligible coordinates whose residual is
  omitted (reset to the reference).  Coordinates with `d = 0` count as omitted
  when selected; the natural zero fraction is reported separately.
- Targets: `P = 0.95` and `P = 0.98`.
- Global selection: exact k smallest scores over all eligible coordinates.
- DACP allocation: Weibull moments per decoder layer.  The original layer cap
  0.95 makes 0.95 degenerate (every layer at the cap) and 0.98 infeasible; the
  cap is set to 0.99 for both targets, and feasibility under 0.95 is logged.
- Payload bytes: per tensor `min(bitmap + 2·kept, 4·kept + 2·kept)`; optimizer
  moments (2 × 2 bytes per trainable parameter) are lossless and reported
  separately.  One archive per run is serialized and decoded bit-exactly.

## Metrics

- Token-weighted mean NLL on validation (32,736 predicted tokens) and on the
  gate set; PPL = exp(NLL).
- ΔLoss_imm = restored − uncompressed at step 200; ΔLoss_final = after 50
  recovery steps (restored moments, same data and rounding noise) minus the
  no-compression branch (two replicates measure the trajectory noise floor).
- Per round: coverage mass (mean, min, p01/p05/p10, median), repeat rate,
  top-1-stratified coverage, forward/backward/sampling/update/selection
  seconds, peak memory per device, mask overlap with HVP and previous round,
  Spearman rank correlation and `u`–`Hd` cosine on a fixed 1/2003 subsample.

## Decision rules fixed before running

- Near-lossless: ΔLoss_imm ≤ log(1.01) nats (≤ 1% relative PPL increase).
- Probe non-inferior to `hvp_global`: `ΔLoss_probe − ΔLoss_HVP ≤ 0.002` nats
  for both immediate and final loss in every seed (3 probe seeds 0, 1, 2).
- Saturation at m: for every m' ≥ m (up to 8), seed-mean ΔLoss_imm(m') is within
  0.0005 nats of seed-mean ΔLoss_imm(8).
- Repeated shocks: 3 compression–recovery cycles of 50 steps each; the
  reference of cycle c+1 is the restored checkpoint of cycle c.

## Baselines

`excp_style` (|d|) and `inshrinkerator_style` (|g ⊙ θ|) are scoring-stage
transfers with exact global selection, identical residual, recovery and
sparsity.  They exclude the original systems' quantization, optimizer-state
compression, search and encoding, so they are not full reproductions of
ExCP (ICML 2024) or Inshrinkerator (SoCC 2024).  `first_order` (|g ⊙ d|)
isolates the value of the curvature term.

## Deviations

- `prepare` ran from commit 3394680, deployed with CRLF line endings (git
  archive on Windows); `score`, `branches` and `chains` ran from 2f80912,
  whose deployed files were verified byte-identical to the git blobs.  The
  prepare code path is unchanged between the two commits apart from line
  endings.
- A 32B smoke test (2 HVP blocks, 2 probe rounds, separate work directory,
  results discarded) showed that 0.18% of residual entries are not exactly
  undoable in bf16 and that pinning host memory took minutes; 2f80912 fixed
  both before any confirmatory run.  No tolerance, split, seed or rule changed.
- The multi-cycle probe method uses seeds `100·cycle` (fixed in code before
  the run), not the cycle-1 seeds 0–2.
- The P = 98% multi-cycle runs were stopped after `dacp_0.98` cycle 2 to free
  the GPUs for the diagnostic and sweep; no 98% chain result is reported.
- Added after the fixed protocol (exploratory, not confirmatory): the exact
  coverage diagnostic (`run_coverage_diagnostic_32b.py`) and the
  immediate-restoration sweep (`run_probe_sweep_32b.py`; sparsities 0.5–0.995,
  references step 100 and base weights, Weibull cap 0.999 because 0.99 cannot
  reach 99.5%).  The first sweep launch (6f102b5) stopped at that cap; results
  come from c482bce.
- Small-model breadth runs (Pythia-410M/Alpaca) used a Python 3.10 environment
  assembled on the host (`experiments/scripts/remote/build_py310_env.sh`),
  because the repository requires Python ≥ 3.10; the checkpoint was produced by
  `finetune_pythia_410m.py` with its defaults on GPU 7.
