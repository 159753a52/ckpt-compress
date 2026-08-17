<!-- created: 2026-08-17 -->

# Damage Surrogate Validation

This experiment measures mask-level correlation between a residual score cost and the immediate loss increase caused by restoring a masked residual. It makes no per-parameter claim and reports only aggregate mask-level rank agreement.

For each candidate mask, the validation logic computes the sum of non-negative score values at pruned coordinates. It then evaluates the same fixed evaluation batches after constructing:

```text
restored = previous_reconstructed + keep_mask * residual_delta
actual_delta_loss = masked_loss - uncompressed_current_loss
```

The default run contains 32 equal-budget masks: magnitude, first-order, Taylor, and 29 deterministic random masks. The random masks use seed 42 and the Taylor mask's per-layer prune counts. Spearman and Kendall rank correlations are reported separately for the magnitude, first-order, and Taylor score families.

## Defaults

- workload: `gpt2_medium_wikitext103`
- seed: `42`
- prune ratio: `0.5` over eligible residual coordinates
- recovery count: `K=1`
- mask count: `32`
- device: `cuda` for the full run; `--dry-run` does not load a model or GPU

The current state is constructed with the existing manifest checkpoint, repeated-seed batch partition, optimizer restoration, scheduler protocol, and one pre-recovery `train_segment`. No alternate training schedule is introduced here.

## Output

The JSON is written with the shared atomic `write_json` helper. A non-dry-run invocation refuses to overwrite an existing output path before loading the checkpoint, data, model, or querying CUDA. A rerun must use a new `--output` path. In dry-run mode, `output_exists` is reported in the plan and no file is changed. The JSON records:

- manifest, checkpoint, and source digests;
- evaluation batch count and hash;
- device, model dtype, seed, workload, prune ratio, and `K`;
- recovery state-construction and scoring batch hashes;
- mask-generation configuration, including exact target and Taylor per-layer counts;
- one record per mask with `id`, `source`, `pruned_count`, `predicted_cost`, `predicted_costs`, `base_loss`, `masked_loss`, and `actual_delta_loss`;
- Spearman and Kendall statistics for every score family.

The validator rejects non-finite values, negative score values, unequal prune counts, duplicate masks, and non-finite rank statistics. After every mask it checks that weights, parameter versions, module modes, `requires_grad`, and original gradients are restored.

## Commands

Plan only:

```powershell
python experiments/scripts/run_damage_surrogate_validation.py --dry-run
```

GPU run:

```powershell
python experiments/scripts/run_damage_surrogate_validation.py --device cuda --output results/diagnostics/damage_surrogate_validation.json
```
