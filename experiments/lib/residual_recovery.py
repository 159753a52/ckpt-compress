"""Reusable residual checkpoint recovery and allocation diagnostics.

This diagnostic computes one block-diagonal HVP at the current checkpoint and
reuses the resulting residual Taylor scores for four masks:

* residual magnitude + uniform per-layer allocation
* residual Taylor score + uniform per-layer allocation
* residual Taylor score + two-parameter Weibull MoM allocation
* residual Taylor score + exact empirical global threshold

By default no optimizer is constructed and no continuation training is
performed.  An optional, explicitly counted matched continuation can be run
after scoring.  A keep mask is applied to the checkpoint residual, not to the
full model weight.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.residual_allocation import (  # noqa: E402
    bounded_largest_remainder_counts,
    budget_tangent_dct_directions,
    calibrate_quantile_smooth_allocation,
    directional_layer_counts,
    fit_layer_weibull_mom,
    fit_weibull_from_moments,
    fit_weibull_mom,
    largest_remainder_counts,
    reconstruct_directional_gradient,
    trust_region_counts,
    uniform_counts,
    weibull_cdf,
    weibull_counts,
)
from experiments.lib.residual_calibration import (  # noqa: E402
    batch_loss_values,
    calibrate_spectral_allocation,
    calibrate_trust_region_allocation,
)
from experiments.lib.residual_masks import (  # noqa: E402
    MaskDict,
    TensorDict,
    apply_layer_mask,
    apply_mask_from_device_states,
    cache_mask_states_on_device,
    exact_keep_mask,
    exact_keep_mask_from_order,
    global_mask,
    layer_mask_at_count,
    layer_masks,
    layer_rates,
    layer_score_orders,
    mask_metrics,
    mask_overlap,
    restore_with_mask,
)
from experiments.lib.residual_runtime import (  # noqa: E402
    LoadedTrainingCheckpoint,
    batch_hash,
    checkpoint_optimizer_state,
    checkpoint_state,
    configure_hf_offline,
    empty_device_cache,
    evaluate_lm,
    evaluate_task,
    lm_loss,
    load_token_batches,
    load_training_checkpoint,
    optimizer_state_to_cpu,
    peak_memory_bytes,
    reset_peak_memory,
    set_seed,
    sha256_file,
    synchronize_device,
    task_loss,
    write_json,
)
from experiments.lib.residual_training import (  # noqa: E402
    RepeatedSeedBatchPlan,
    SeedBatchPartition,
    build_optimizer,
    clone_model_state_to_cpu,
    partition_seed_batches,
    partition_repeated_seed_batches,
    seeded_training_batches,
    train_segment,
)
from experiments.lib.residual_scoring import (  # noqa: E402
    compute_block_first_order_scores,
    compute_block_taylor_scores,
    eligible_layers,
    model_checksum,
    transformer_layers,
)
from experiments.lib.residual_short_config import (  # noqa: E402
    parse_args,
    validate_short_config,
)
from experiments.lib.residual_short_experiment import (  # noqa: E402
    continue_training,
    main,
)
from experiments.lib.residual_short_methods import (  # noqa: E402
    build_short_gate_masks,
    build_short_residual_scope,
    diagnose_short_gate_masks,
)


if __name__ == "__main__":
    main()
