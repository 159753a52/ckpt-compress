# V100 residual allocation gate

This directory contains short, one-shot diagnostics. All reported metrics were
produced by the checked-in scripts; no values are synthesized.

## Internally matched recovery gate

The decisive run is `20260809_105909_p0.50.json`.

- Start from GPT-2 Medium checkpoint step 1000.
- Train 20 explicit steps (`batch_size=2`, `seq_length=128`, `lr=5e-5`) to
  produce the step-1020 recovery snapshot.
- Compute residual Taylor scores in evaluation mode from five batches. No
  optimizer exists during scoring, and pre/post parameter checksums are equal.
- Prune exactly 150,994,944 of 301,989,888 eligible residual parameters (50%).
- Evaluate all masks on the same 50 batches (12,800 tokens).
- Continue every restored snapshot for 20 explicit steps with the same AdamW
  state, data, learning rate, and random seed, then evaluate again. This makes
  comparisons within the gate paired, but it does not exactly reproduce the
  older main experiment, which used a cosine learning-rate schedule.

| Method | Immediate PPL | PPL after 20 steps |
|---|---:|---:|
| No compression | 24.1449 | 29.2762 |
| Residual magnitude + uniform | 23.6530 | 29.0799 |
| Taylor + uniform | **23.2458** | **28.9950** |
| Taylor + Weibull MoM | 23.2414 | 29.0789 |
| Taylor + exact global threshold | 23.2749 | 29.1372 |

Taylor + uniform improves over residual magnitude by 0.4072 PPL (1.72%)
immediately and 0.0849 PPL (0.29%) after matched continuation. Weibull improves
over uniform Taylor by only 0.0045 PPL immediately and is 0.0840 PPL worse
after continuation. This is one-seed evidence for residual Taylor scoring
within this gate, but it does not support claiming that Weibull allocation is
the main quality contribution.

Both uncompressed training segments are harmful on the fixed evaluation slice:
PPL rises from 20.3712 at step 1000 to 24.1449 after preparation and 29.2762
after continuation. Compression partially rolls back those harmful updates.
The result therefore cannot be interpreted as normal-training
restoration-induced degradation or as a strict reproduction of the old main
table.

The five-batch HVP took 15.0 seconds and 2.98 GiB peak allocated GPU memory.
The full gate took 121.3 seconds on one Tesla V100 32GB. The scoring checksum
delta was exactly zero and no parameter version changed.

## Exploratory storage-delta gates

The earlier files use the fixed step-800 to step-1000 residual rather than the
paper's 20-step recovery protocol. They are retained to make the search trace
complete.

| Result | Magnitude + uniform | Taylor + uniform | Taylor + Weibull | Exact global |
|---|---:|---:|---:|---:|
| `20260809_104327_p0.30.json` | 18.5463 | 18.5451 | 18.5451 | 18.5452 |
| `20260809_104626_p0.50.json` | 18.5449 | 18.5431 | 18.5431 | 18.5432 |
| `20260809_105053_p0.50.json` (20-step continuation) | 26.7765 | 26.8511 | 26.9292 | 27.0046 |

These exploratory gates do not show a meaningful advantage and must not be
used as positive paper evidence.

## Reproduction

Generate the short recovery checkpoint:

```bash
python experiments/scripts/prepare_short_recovery_checkpoint.py \
  --steps 20 --batch-size 2 --seq-length 128 \
  --learning-rate 5e-5 --seed 42
```

Run the matched gate:

```bash
python experiments/scripts/run_v100_allocation_gate.py \
  --reference-checkpoint checkpoints/gpt2_medium_wikitext103_1000steps/checkpoint_step_1000.pt \
  --current-checkpoint checkpoints/gpt2_medium_wikitext103_short_recovery/checkpoint_step_1020.pt \
  --prune-ratio 0.50 --batch-size 2 --seq-length 128 \
  --eval-batches 50 --hvp-batches 5 --train-batch-offset 20 \
  --continuation-steps 20 --continuation-lr 5e-5 --seed 42
```

The generated 4.46 GB checkpoint is intentionally ignored by Git. Its small
provenance JSON records both checkpoint hashes and the exact training-batch
hash.

## Scope

This is a one-seed short gate over 12,800 evaluation tokens, not a paper-ready
statistical result. It uses constant learning rate while the old main result
used cosine decay. The Weibull allocator uses a per-layer cap of 0.8 while the
exact global diagnostic is uncapped, so their proxy regrets are not the same
constrained optimization problem. Exact additive-score optimality also does
not imply optimal actual loss; this is visible in the measured PPL.
