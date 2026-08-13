# Paper runner complexity and measurement

This note describes the declaration-driven runner in
`experiments/scripts/run_paper_experiments.py`. It intentionally does not attach
wall-clock estimates to models that have not been measured on the target
machine. The raw job JSON is the source of truth for timing.

## Symbols

| Symbol | Meaning |
|---|---|
| `P` | eligible residual parameters |
| `L` | structural layers |
| `B` | scoring batches per recovery (`hvp_batches`) |
| `T` | matched recovery training steps (`total_steps`) |
| `E` | evaluation batches |
| `K` | recovery events |
| `M` | methods in the selected suite |
| `S` | seeds in the selected job |

Every `(workload, prune_ratio, K)` job runs all selected methods and seeds with
the same checkpoint, batch plan, and evaluation batches. A workload context is
loaded once and reused across its pending jobs.

## Per-method work

`no_compression` performs `T` training steps and `K + 1` evaluations. It does
not score or mask residuals.

`excp_style` adds one residual-magnitude pass and one exact global selection per
recovery. The implementation is linear in `P`; its histogram selection uses
fixed-size score chunks and does not concatenate all eligible scores.

`inshrinkerator_style` adds first-order scoring over `B` batches, plus the same
linear exact global selection per recovery.

`dacp` adds blockwise Taylor/HVP scoring over `B` batches, per-layer moment
collection, Weibull allocation, and exact per-layer masks. Allocation metadata
records the exact prune target and layer rates.

Ignoring model-dependent forward/backward constants, one job therefore has the
following accounting shape:

```text
training steps     = M * S * T
evaluations        = M * S * (K + 1)
compression events = (M - 1) * S * K
scoring batches    = S * K * B for each gradient-based method
```

These counts are planning quantities, not runtime predictions. HVP cost,
sequence length, model architecture, dtype, and device memory pressure dominate
wall time.

## Memory boundaries

The workload context holds one decoded checkpoint, one training pool, and one
evaluation cache. A method trajectory additionally holds one model, optimizer
state, reconstructed/current CPU model states, and method-specific scores.

Global exact selection maps float32 scores to monotonic integer keys and uses
two 65,536-bin histograms. Non-contiguous tensors are sliced recursively before
conversion, so any reshape copy is bounded by the configured score chunk rather
than the full tensor. The final boolean mask remains `O(P)` because it is the
experiment output needed to apply recovery.

DACP's blockwise scorer bounds temporary autograd work to one structural block,
but model state, optimizer state, residual states, and final masks still scale
with the model. A successful CPU smoke test is not evidence that a full model
fits a particular GPU.

## Runtime evidence

Each result record stores:

- `wall_seconds` for the complete method trajectory;
- `training.seconds` for each recovery segment and final segment;
- scoring/allocation timings inside compression metadata where available;
- model, dataset, seed, sparsity, `K`, checkpoint digest, and batch identities.

Compare timings only when those identities and the hardware/software
environment match. Do not copy estimates from the retired Table/Gamma runners
into the paper. Aggregate measured JSON records after the first GPU gate, and
report the device, dtype, batch size, sequence length, and number of repetitions
beside any runtime claim.

## Planning the declared suite

Use the manifest dry-run to enumerate exact jobs without loading a model:

```bash
python experiments/scripts/run_paper_experiments.py --dry-run
```

For a smaller gate, select one workload, ratio, recovery count, and seed. The
selection is accepted only when every value is declared in
`experiments/configs/paper_experiments.yaml`.
