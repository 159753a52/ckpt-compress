# Full-weight pruning-only comparison

This protocol is a single-seed (`42`) mechanism comparison on the same GPT-2
Medium checkpoint. It prunes complete eligible matrix weights directly and
compares:

- `dacp_fullweight`: native Taylor-HVP scoring with the current full weight
  `theta` as the probe, followed by Weibull-moment allocation.
- `inshrinkerator_style_fullweight`: style-only searched magnitude or
  first-order scoring with per-type allocation projected to the same exact
  global budget.

Both methods use the same eligible parameter names, checkpoint state, scoring
batches, evaluation batches, target count, and direct-zeroing application. The
comparison does not reconstruct a residual and does not continue training.

The Inshrinkerator-style result is deliberately not a full Inshrinkerator
reproduction. It does not call `InshrinkeratorCompressor` and does not include
protection, quantization, byte packing, or encoding. The only valid paper
label is **Inshrinkerator-style full-weight pruning**. A claim about the full
Inshrinkerator compressor or byte-level wins requires a separate experiment
with its actual quantization and encoding pipeline and matched byte budgets.

The search JSON is provenance-bound to the model, dataset, seed, checkpoint
digest, source digest, and scoring/evaluation batch digests. Old search JSON
files without those fields fail closed. `protect_fraction` is recorded as a
search/config provenance field; this pruning-only comparison does not apply a
protection step.

## Formal run

First produce the searched per-type evidence. Use one epsilon so the output
path matches the formal manifest:

```bash
python experiments/scripts/run_inshrinkerator_like_search.py \
  --model gpt2-medium \
  --dataset wikitext103 \
  --checkpoint checkpoints/gpt2_medium_wikitext103_1000steps/checkpoint_step_1000.pt \
  --data-dir data/wikitext103 \
  --epsilons 0.05 \
  --num_steps 8 \
  --hvp_batches 8 \
  --eval_batches 100 \
  --batch_size 2 \
  --seq_length 128 \
  --seed 42 \
  --protect-fraction 0.005 \
  --device cuda \
  --output-dir results/paper_runs/inshrinkerator_search
```

Then inspect and run the formal comparison:

```bash
python experiments/scripts/run_paper_fullweight_comparison.py \
  --manifest experiments/configs/paper_fullweight_inshrinkerator.yaml \
  --dry-run

python experiments/scripts/run_paper_fullweight_comparison.py \
  --manifest experiments/configs/paper_fullweight_inshrinkerator.yaml \
  --device cuda \
  --output-dir results/paper_runs/fullweight_inshrinkerator
```

The comparison writes `suite_manifest.json` and one atomic JSON per pruning
ratio. Resume requires the same plan and provenance digests and skips only
complete ratio records whose stored file digest still matches the suite.

## Timing boundary

Each method record reports `scoring_wall_seconds`,
`allocation_wall_seconds`, `evaluation_wall_seconds`, and
`evaluation_peak_gpu_memory_bytes` separately. The ratio-level
`allocation_and_evaluation_wall_seconds` is not an end-to-end compressor
runtime: it does not merge shared scoring time and it does not represent the
actual Inshrinkerator byte-packing/encoding pipeline. These results therefore
support mechanism and quality comparisons, not a complete runtime or
compression-ratio claim for the full compressor.
