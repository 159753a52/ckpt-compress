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
jobs and their SHA-256 digests. The runner refuses missing checkpoints and
undeclared workload/ratio/K/seed selections.

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

## Figure provenance

`generate_paper_figures.py` does not generate synthetic evidence. Distribution
validation JSON must include empirical and fitted CDF grids. Allocation figures
read actual DACP `layer_rates` from a completed, digest-verified paper suite.

```bash
python experiments/scripts/generate_paper_figures.py \
  --fit-results results/paper_results/gamma_validation/<fit_results.json> \
  --suite results/paper_runs/<suite>/suite_manifest.json \
  --output-dir ../paper/figs
```

Every generated PDF/PNG has a sibling `.provenance.json` containing source
paths and hashes. Missing, incomplete, malformed, or modified inputs fail
instead of falling back to simulated data.
