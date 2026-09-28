# Pythia-410M / Alpaca campaign report (2026-09-28, ~06:10-10:05 UTC)

All numbers below are copied from the summary files in this directory, which
are generated from the runner / driver JSON files by
`experiments/scripts/summarize_probe_pythia410m_campaign.py` and
`experiments/scripts/summarize_system_bytes.py`. Style baselines
(`excp_style`, `inshrinkerator_style`) are matched-protocol scoring rules, not
reproductions of the ExCP or Inshrinkerator systems.

## Status per campaign item

| item | suite dir | status at report time |
|---|---|---|
| LR fix | `lr_pilot`, `lr_pilot2` | complete; lr 1e-6 chosen |
| 1. ratio sweep K=1 | `main_k1` | complete: 6 ratios x 3 seeds x 9 methods (162 records) |
| 2. K in {3,5} | `repeated_k3_k5` | p0.9 K=3 and K=5 complete (3 seeds); p0.98 K=3 partial; p0.98 K=5 not started |
| 3. step-3000 checkpoint | (not yet run) | queued on GPU 7 after item 2 |
| 4. system bytes | `system_bytes` | complete: 7 ExCP alphas x 3 seeds x 3 methods; Inshrinkerator-like 4 configs x 1 seed |

The GPU-7 queue is still running (item 2 remainder, then the 3000-step
fine-tune, its reference evaluation, and the step-3000 suite). Remote outputs:
`/home/gridtest/dacp_probe/campaign/{repeated_k3_k5,finetune_step3000,step3000_k1}`;
pull with `launch/pull.ps1 "repeated_k3_k5 finetune_step3000 step3000_k1"`.
The step-3000 checkpoint (4.9 GB) will be written to
`/home/gridtest/dacp_probe/assets/checkpoints/pythia_410m_alpaca_step3000/`.

## Recovery learning rate

Manifest `experiments/configs/probe_pythia410m_campaign.yaml` (new; existing
manifests untouched) uses lr 1e-6. Pilots (`no_compression`, seed 42, K=1;
checkpoint PPL 5.5910 on the same 100 eval batches; values = PPL after the
first 50-step segment / final, from `lr_pilot2/summary_lr_pilots.md` and the
earlier `probe_small_models` run for 5e-5):

| lr | after segment 1 | final |
|---|---|---|
| 5e-5 | 10.24 | 8.59 |
| 2e-5 | 6.6304 | 6.3720 |
| 1e-5 | 5.9606 | 5.8690 |
| 5e-6 | 5.7205 | 5.6989 |
| 2e-6 | 5.6170 | 5.6120 |
| 1e-6 | 5.5949 | 5.5928 |

Every rate, including 1e-5 and 2e-5, raises the control's perplexity (recovery
uses batch 2 x 128 tokens vs the fine-tune's 4 x 512, and the fine-tune ends at
lr ~0). 1e-6 is the largest tested rate at which the control does not degrade;
over 3 seeds the control's final PPL is 5.5877 +- 0.0041 (`main_k1`).
Consequence: the recovery residual is tiny, so the K=1/K>1 differences between
methods are at the 1e-3 PPL level (see below). Other manifest changes:
`max_layer_ratio` 0.995 (the loader requires it to be >= every prune ratio, so
0.98/0.99 need it), `train_pool_batches` 140 (= 100 + 5 x 8, so K=1/3/5 share
one cached pool). dacp ran at 0.99 without failures.

## 1. Main ratio sweep, K=1 (`main_k1/summary.md`)

Final PPL mean over 3 seeds; d_nc = paired mean difference vs no_compression
(no_compression = 5.5877 +- 0.0041); sign counts are wins/ties/losses over seeds.

| prune | excp_style | insh_style | dacp | probe_topk_m1 | dacp vs excp | dacp vs insh |
|---|---|---|---|---|---|---|
| 0.50 | 5.5870 | 5.5853 | 5.5864 | 5.5864 | 3/0/0 | 0/0/3 |
| 0.70 | 5.5859 | 5.5856 | 5.5857 | 5.5857 | 2/0/1 | 2/0/1 |
| 0.90 | 5.5853 | 5.5869 | 5.5857 | 5.5857 | 0/0/3 | 3/0/0 |
| 0.95 | 5.5860 | 5.5874 | 5.5862 | 5.5862 | 1/0/2 | 3/0/0 |
| 0.98 | 5.5871 | 5.5879 | 5.5869 | 5.5869 | 2/0/1 | 3/0/0 |
| 0.99 | 5.5877 | 5.5881 | 5.5872 | 5.5873 | 3/0/0 | 3/0/0 |

All compressed methods except inshrinkerator_style at 0.98/0.99 end at or below
the control (compression discards part of an update that slightly hurts eval
PPL). first_order, taylor, dacp and all probe variants are within 1e-4 of each
other at every ratio. Differences to the baselines are 1e-4 to 2e-3 PPL, i.e.
comparable to the seed std (~1e-3); they are consistent in sign across seeds in
some cells but should not be presented as a meaningful quality gap.

## 2. Repeated recoveries (`repeated_k3_k5/summary.md`)

p=0.9, final PPL (3 seeds): K=3: no_compression 5.5884, excp_style 5.5862,
inshrinkerator_style 5.5877, dacp 5.5866 (vs excp 0/0/3, vs insh 3/0/0).
K=5: no_compression 5.5867, excp_style 5.5852, inshrinkerator_style 5.5872,
dacp 5.5855 (vs excp 0/0/3, vs insh 3/0/0). taylor/probe variants match dacp to
1e-4. p=0.98 K=3 is partial (2 seeds for baselines, 1 for dacp/probes).

## 4. System-level bytes vs restored PPL (`system_bytes/summary.md`)

Setting: Pythia-410M step-1000 checkpoint, residual against the pretrained
base (PPL 14.4560), payload = weight residual + exp_avg + exp_avg_sq; restored
PPL of the decoded weights on the same eval batches (no recovery training).
fp32 reference: 4864.0 MB. Bytes = torch.save payload after zlib level 6
(lzma preset 1 also in the summary). Fidelity labels:

- `excp_pipeline`: reimplementation of the ExCP pipeline following the
  authors' released `baselines/excp/upstream/compress_pythia.py` (per-tensor
  threshold median(|dW|)*min(2.5, alpha/sqrt(mean v)), joint moment rule with
  beta=2, 4-bit non-zero K-means, index 0 = zero, 2 codes/byte, fp16 codebooks);
  faiss K-means replaced by a seeded 1-D Lloyd K-means on a faiss-sized sample.
- `dacp_excpq`, `probe_topk_m1_exact_global_excpq`: same pipeline, but the
  surviving residual entries on eligible Transformer matrices are chosen by our
  scores at exactly ExCP's eligible sparsity; moments follow the ExCP rule
  with our mask; same quantizer.
- `inshrinkerator_like`: our simplified reimplementation (repository
  partition(), bf16 protection, sign + magnitude K-means) of full weights and
  moments; no sketch K-means, no cross-checkpoint delta/RLE (single checkpoint).
  One seed only.

| alpha | eligible sparsity | ExCP PPL / MB | DACP PPL / MB | probe_topk PPL / MB | DACP vs ExCP (PPL, bytes) |
|---|---|---|---|---|---|
| 1e-5 | 0.0121 | 5.5763 / 271.41 | 5.5754 / 271.53 | 5.5748 / 271.92 | 3/0/0, 1/0/2 |
| 5e-5 | 0.0604 | 5.5763 / 271.77 | 5.5706 / 273.14 | 5.5706 / 273.32 | 3/0/0, 0/0/3 |
| 2e-4 | 0.2374 | 5.5727 / 246.11 | 5.5453 / 250.72 | 5.5456 / 250.65 | 3/0/0, 0/0/3 |
| 4e-4 | 0.4492 | 5.5465 / 206.10 | 5.5134 / 212.92 | 5.5138 / 212.80 | 3/0/0, 0/0/3 |
| 6e-4 | 0.6197 | 5.5325 / 167.23 | 5.5274 / 176.86 | 5.5250 / 176.86 | 3/0/0, 0/0/3 |
| 1e-3 | 0.8227 | 5.8906 / 102.92 | 5.8635 / 114.66 | 5.8496 / 114.73 | 3/0/0, 0/0/3 |
| 1e-2 | 0.9018 | 6.5616 / 76.56 | 6.6610 / 81.53 | 6.6533 / 81.60 | 0/0/3, 0/0/3 |

Inshrinkerator-like (seed 42): 16 bins, prune 0: 5.9669 / 673.71 MB; 64 bins:
5.6175 / 933.26 MB; 254 bins: 5.6010 / 1218.66 MB; 16 bins with the adapter's
default prune 0.2: 733.82 / 657.21 MB.

Reading: at matched weight sparsity DACP and probe_topk give lower restored
PPL than ExCP for alpha <= 1e-3 (3/3 seeds each) but always cost more bytes
(+0.1 to +11.8 MB), because more exp_avg entries pass ExCP's joint rule under
our masks; at 90% (alpha 1e-2, near ExCP's 2.5x median cap) DACP is worse
(+0.099 PPL). A proper Pareto claim needs equal-byte comparison; from these
points DACP at alpha 6e-4 (176.86 MB, 5.5274) is below and cheaper than ExCP at
alpha 4e-4 (206.10 MB, 5.5465), but ExCP at 6e-4 (167.23 MB, 5.5325) is
cheaper with a 0.005 higher PPL. Restored PPL below the checkpoint's 5.5910 at
low sparsity is measured, not an artefact we can explain yet. ExCP's
threshold cap limits its sparsity to ~0.90 eligible on this checkpoint, so the
0.95-0.99 regime has no ExCP operating point. The Inshrinkerator-like points are
dominated here (4-7x vs 18-64x) because it quantizes full weights without a
residual base; its paper's cross-checkpoint delta stage cannot apply.

## Code, commits, deployment

Local branch `codex/probe-coverage` (new commits, all new files):
e286b0f (pilot manifest + `run_campaign_gpu7.sh`), 39a3c76 (second pilot),
3f40f94 (campaign manifest, summary script, step-3000 fine-tune helper),
f864e1b (`run_system_bytes_comparison.py`), 0d39433 (empty alpha list fix),
plus the results commit. No existing library code was modified; full local
test suite: 634 passed, 2 skipped (`python -m pytest -q -p no:cacheprovider -x`).
Remote code dirs `/home/gridtest/dacp_probe_code/<sha7>` were created by
`cp -a` of the previous dir and `git fetch` of a `git bundle <prev>..HEAD`
followed by `git reset --hard <sha>` (the depth-1 `.git` thus holds the exact
commit; runner provenance shows `git_dirty: false` except for the untracked
`DACP_CODE_COMMIT`, which is not a source file). Suites ran from 3f40f94
(`main_k1`, `repeated_k3_k5`, queued items), pilots from e286b0f/39a3c76,
suite 4 from 0d39433.

Exact commands (see `launch/`): GPU queue = `launch/q_main.txt`, executed line
by line by `launch/launch.sh 3f40f94 /tmp/dacp_q_main.txt` (nohup, one process
at a time, CUDA_VISIBLE_DEVICES=7, DACP_GPU_MEM_FRACTION=0.14 via
`gpu7_env.sh`). Suite 4 ran CPU-only (CUDA_VISIBLE_DEVICES="", 24 threads) via
`launch/sys_cpu.sh 0d39433 <name> --seeds S --alphas ... --ours dacp,probe_topk_m1_exact_global --insh-bins=`
and `... insh_p0_s42 --seeds 42 --alphas= --ours= --insh-bins 16,64,254 --insh-prune 0.0`.

## Resources, runtimes, failures

- GPU 7 peak memory of our processes (20 s sampling): pilots 9.0 GB, main_k1
  15.1 GB (cap 0.14 x 143.8 GB = 20.1 GB). The sampler of repeated_k3_k5 recorded
  no samples, so its peak is not measured.
- Runtimes: lr_pilot 142 s, lr_pilot2 212 s, main_k1 6278 s; suite-4 CPU jobs
  3315-5128 s each (peak RSS ~20-22 GB). Remote disk use: ~5 MB results + 5 x
  38 MB code dirs (step-3000 checkpoint will add 4.9 GB).
- Incidents: (1) the first main queue had to be restarted (queue runner
  stdin fix); the old runner briefly started a `repeated_k3_k5` dry-run
  (empty plan.json only) before it was killed; that directory was deleted and
  `main_k1` restarted from scratch. No two of our GPU jobs overlapped in
  training. (2) first suite-4 launch failed on argparse (empty values), relaunched.
  (3) `system_bytes/smoke.*` is a validation run from f864e1b (seed 42, alpha
  1e-3), kept for record and excluded from `summary.md`.
