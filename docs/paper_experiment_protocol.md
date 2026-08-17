# Paper experiment protocol

This repository uses one declaration-driven runner for the five model/task
configurations named in the paper. The objective is to collect the minimum
evidence needed for each claim, not to exhaust every method/configuration
combination.

## Evidence boundaries

The main matched recovery suite exposes four method identities:

| Name | Fidelity | What is compared |
|---|---|---|
| `no_compression` | native | Training without lossy recovery |
| `dacp` | native | Taylor HVP score plus Weibull allocation |
| `excp_style` | style | ExCP residual-magnitude criterion with strong global exact-top-k selection |
| `inshrinkerator_style` | style | Inshrinkerator full-weight first-order criterion with strong global exact-top-k selection |

The two `style` adapters deliberately exclude the papers' complete optimizer,
quantization, search, and encoding pipelines. They are valid for a controlled
scoring-stage comparison, but must not be reported as full ExCP or full
Inshrinkerator reproductions. The manifest claim gate rejects that upgrade.

All methods use the same checkpoint, data hashes, training segments, residual
sparsity, seed, and recovery schedule. Baseline hyperparameters may be tuned on
the same validation budget as DACP; intentionally weakening a baseline is not
an acceptable protocol.

The recovery trajectory restores the checkpoint's model weights and AdamW
moments, then sets the manifest learning rate and starts a fresh cosine schedule
over the declared recovery horizon. It does not claim to continue the scheduler
from the checkpoint's original pretraining or fine-tuning run. Every job records
this policy and the checkpoint step as structured provenance so a resume cannot
silently mix the two interpretations. The runner also rejects a checkpoint whose
self-reported step differs from the workload's declared `checkpoint_step`.

## Minimal run order

1. Validate one paired GPT-2 Medium job at 50% sparsity and `K=1`.
2. Run GPT-2 Medium at `K=3` to validate compound recovery.
3. Run the five-model 50% main table with the declared seeds.
4. Run the GPT-2 Medium 30/50/70/90% sweep only if the first three gates show a
   stable effect.
5. Run distribution validation and any ablation only for claims retained in the
   paper after the main table is available.

Inspect the exact plan before using a GPU:

```bash
python experiments/scripts/run_paper_experiments.py \
  --dry-run \
  --workload gpt2_medium_wikitext103 \
  --prune-ratio 0.5 \
  --recoveries 1 \
  --seeds 42
```

Run the first gate:

```bash
python experiments/scripts/run_paper_experiments.py \
  --workload gpt2_medium_wikitext103 \
  --prune-ratio 0.5 \
  --recoveries 1 \
  --seeds 42 \
  --device cuda \
  --output-dir results/paper_runs/gate1
```

Each job writes incremental raw JSON. `suite_manifest.json` records completed
jobs and their SHA-256 digests. The runner refuses missing checkpoints,
undeclared workload/ratio/K/seed selections, duplicate CLI selections, and
non-finite metrics.

Existing output is never overwritten implicitly. After an interruption, resume
the exact same plan explicitly:

```bash
python experiments/scripts/run_paper_experiments.py \
  --workload gpt2_medium_wikitext103 \
  --prune-ratio 0.5 \
  --recoveries 1 \
  --seeds 42 \
  --device cuda \
  --output-dir results/paper_runs/gate1 \
  --resume
```

Resume validates the manifest, source tree fingerprint (including untracked and
deleted Python/YAML/TOML/shell sources), checkpoint digest, training/evaluation
batch counts and hashes, method contracts, exact allocation target, job
configuration, and stored batch plans. A `no_compression` cycle must preserve
its before/after metrics exactly. Only complete, validated `(seed, method)`
records are skipped. A changed source tree, plan, checkpoint, batch plan, or
result record fails instead of mixing incompatible evidence.

## Single-seed 2x2 ablation

`paper_ablation_2x2.yaml` is an independent manifest for the GPT-2 Medium,
50% pruning, `K=1`, seed `42` gate. Its five methods form a 2x2 scoring and
allocation comparison, with `dacp` as the Taylor-HVP plus Weibull-moment cell
and `taylor_exact_global` as an exact global top-k oracle:

| Method | Score | Allocation |
|---|---|---|
| `magnitude_uniform` | residual magnitude | uniform per layer |
| `magnitude_weibull` | residual magnitude | Weibull moments |
| `taylor_uniform` | Taylor HVP | uniform per layer |
| `dacp` | Taylor HVP | Weibull moments |
| `taylor_exact_global` | Taylor HVP | exact global top-k oracle |

This isolates scoring and allocation choices only. One seed does not provide
statistical significance and must not be presented as a significance claim.
The manifest is intentionally separate from the default five-workload main
table and has no baseline claim gate to bypass the main-table style-baseline
gate.

Inspect the exact ablation plan without loading a checkpoint:

```bash
python experiments/scripts/run_paper_experiments.py \
  --manifest experiments/configs/paper_ablation_2x2.yaml \
  --methods magnitude_uniform,magnitude_weibull,taylor_uniform,dacp,taylor_exact_global \
  --dry-run
```

Run the fixed seed on the GPU only after the dry-run and input checks pass:

```bash
python experiments/scripts/run_paper_experiments.py \
  --manifest experiments/configs/paper_ablation_2x2.yaml \
  --device cuda \
  --output-dir results/paper_runs/ablation_2x2
```

When all selected jobs for a workload are already complete, resume re-hashes
the checkpoint before skipping model and dataset loading. If a workload has any
pending job, the shared workload context performs that check once while loading
the pending inputs. The stored checkpoint step is protected by that file
digest; the runner never infers or fabricates an unavailable scheduler state.

Paths in the manifest are repository-relative. `--checkpoint-root` and
`--data-root` remap them for another machine. Local Hugging Face model and
tokenizer snapshots default to `data/models`; set `DACP_MODEL_ROOT` to override
that location, or `DACP_DATA_ROOT` to relocate the complete data hierarchy.
`ALPACA_LOCAL_PATH` may point to one JSON file, otherwise the loader checks
`<data-root>/alpaca/alpaca_data.json` and `<data-root>/alpaca_data.json` before
using the Hugging Face dataset source.

Training checkpoints include optimizer state and are decoded with PyTorch's
restricted tensor-only loader. Checkpoints containing custom pickle globals are
rejected. The runner also hashes the checkpoint and locks that digest for every
job of the same workload; the digest records provenance but is not used as a
substitute for the restricted loader.

## Distributed moments

The low-level distributed statistics implementation performs real collectives
for per-block count, sum, squared sum, zero count, and maximum. It records the
backend, world size, and actual communication as `5L` scalars per rank. The
paper runner does not expose this as a complete distributed mode yet: a real
parameter-shard ownership and mask-coordination contract is still required.

The current manuscript also states one `2L`-scalar `allreduce`. That exact
constant is not the current implementation contract and must be revised before
submission, unless a two-moment implementation is introduced together with an
explicit equal-count/data-sharding contract.

## Complexity and timing

[`complexity_analysis.md`](complexity_analysis.md) derives work counts for the
current manifest runner and identifies the timing fields recorded in raw JSON.
It intentionally contains no unverified per-model hour estimates. Runtime
claims must be aggregated from compatible result records produced on the target
hardware.

## Figure provenance

`generate_paper_figures.py` does not generate synthetic evidence. Distribution
validation JSON must include empirical and fitted CDF grids. Allocation figures
read actual DACP `layer_rates` from a completed, digest-verified paper suite.

```bash
python experiments/scripts/generate_paper_figures.py \
  --fit-results results/paper_results/gamma_validation/<fit_results.json> \
  --suite results/paper_runs/<suite>/suite_manifest.json \
  --output-dir ../../paper/figs
```

Every generated PDF/PNG has a sibling `.provenance.json` containing source
paths and hashes. Missing, incomplete, malformed, or modified inputs fail
instead of falling back to simulated data.
