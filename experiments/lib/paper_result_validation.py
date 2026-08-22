"""Schema and evidence validation for paper experiment result JSON."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from experiments.lib.data import get_task_type
from experiments.lib.models import get_model_type
from experiments.lib.residual_protocol import (
    RESIDUAL_MAGNITUDE_UNIFORM_METHOD,
    RESIDUAL_MAGNITUDE_WEIBULL_MOM_METHOD,
    TAYLOR_EXACT_GLOBAL_METHOD,
    TAYLOR_MEANABS_UNIFORM_METHOD,
    TAYLOR_MEANABS_WEIBULL_MOM_METHOD,
    TAYLOR_SIGNED_EXACT_GLOBAL_METHOD,
    TAYLOR_SIGNED_FIRST_ORDER_UNIFORM_METHOD,
    TAYLOR_SIGNED_SECOND_ORDER_UNIFORM_METHOD,
    TAYLOR_SIGNED_UNIFORM_METHOD,
    TAYLOR_SIGNED_WEIBULL_MOM_METHOD,
    TAYLOR_UNIFORM_METHOD,
    TAYLOR_WEIBULL_MOM_METHOD,
)

SCHEMA_VERSION = 5
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_WEIBULL_INVALID_REASONS = {
    "empty values",
    "non-positive maximum",
    "degenerate moments",
    "degenerate coefficient of variation",
    "shape root not bracketed",
    "invalid scale",
}
_CANONICAL_COMPRESSION_METHODS = {
    "residual_magnitude_exact_global": ("residual_magnitude", "exact_global"),
    RESIDUAL_MAGNITUDE_UNIFORM_METHOD: ("residual_magnitude", "uniform_per_layer"),
    RESIDUAL_MAGNITUDE_WEIBULL_MOM_METHOD: ("residual_magnitude", "weibull_moment"),
    "first_order_exact_global": ("first_order", "exact_global"),
    TAYLOR_UNIFORM_METHOD: ("taylor_hvp", "uniform_per_layer"),
    TAYLOR_WEIBULL_MOM_METHOD: ("taylor_hvp", "weibull_moment"),
    TAYLOR_EXACT_GLOBAL_METHOD: ("taylor_hvp", "exact_global"),
}
_REQUIRED_CANONICAL_COMPRESSION_METHODS = {
    RESIDUAL_MAGNITUDE_UNIFORM_METHOD,
    RESIDUAL_MAGNITUDE_WEIBULL_MOM_METHOD,
    TAYLOR_UNIFORM_METHOD,
    TAYLOR_EXACT_GLOBAL_METHOD,
}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value is not allowed: {value}")


def _load_json(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load experiment state from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment state must be a JSON object: {path}")
    _ensure_finite_json(payload, context=str(path))
    return payload


def _ensure_finite_json(value: object, *, context: str) -> None:
    """Reject overflowed JSON numbers such as ``1e999`` anywhere in state."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite JSON value is not allowed in {context}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _ensure_finite_json(child, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _ensure_finite_json(child, context=f"{context}[{index}]")


def _canonical_json_object(payload: Mapping[str, object]) -> dict[str, object]:
    """Normalize tuples and mapping implementations to their on-disk JSON form."""
    serialized = json.dumps(payload, allow_nan=False, sort_keys=True)
    normalized = json.loads(serialized, parse_constant=_reject_json_constant)
    if not isinstance(normalized, dict):
        raise ValueError("Experiment template must be a JSON object")
    return normalized


def _require_equal(
    actual: Mapping[str, object],
    key: str,
    expected: object,
    *,
    context: str,
) -> None:
    if actual.get(key) != expected:
        raise ValueError(f"{context} is incompatible: {key} does not match")


def _require_provenance_compatible(
    actual: object,
    expected: object,
    *,
    keys: Sequence[str],
    context: str,
) -> None:
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        raise ValueError(f"{context} has invalid provenance")
    _validate_source_provenance(actual, context=f"{context} provenance")
    _validate_source_provenance(expected, context=f"{context} expected provenance")
    for key in keys:
        value = actual.get(key)
        if key == "checkpoint_step":
            valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else:
            valid = isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None
        if not valid or value != expected.get(key):
            raise ValueError(f"{context} is incompatible: provenance {key} does not match")


def _validate_source_provenance(value: Mapping[str, object], *, context: str) -> None:
    manifest = value.get("manifest")
    commit = value.get("git_commit")
    dirty = value.get("git_dirty")
    dirty_paths = value.get("dirty_source_paths")
    source_file_count = value.get("source_file_count")
    if not isinstance(manifest, str) or not manifest:
        raise ValueError(f"{context}.manifest must be a non-empty path")
    _require_sha256(value.get("manifest_sha256"), context=f"{context}.manifest_sha256")
    _require_sha256(value.get("source_state_sha256"), context=f"{context}.source_state_sha256")
    if not isinstance(commit, str) or _GIT_OBJECT_PATTERN.fullmatch(commit) is None:
        raise ValueError(f"{context}.git_commit must be a Git object ID")
    if not isinstance(dirty, bool):
        raise ValueError(f"{context}.git_dirty must be boolean")
    if (
        not isinstance(dirty_paths, list)
        or not all(isinstance(path, str) and path for path in dirty_paths)
        or dirty_paths != sorted(set(dirty_paths))
        or any("\\" in path for path in dirty_paths)
    ):
        raise ValueError(f"{context}.dirty_source_paths must be sorted unique relative paths")
    if dirty != bool(dirty_paths):
        raise ValueError(f"{context}.git_dirty disagrees with dirty_source_paths")
    if (
        isinstance(source_file_count, bool)
        or not isinstance(source_file_count, int)
        or source_file_count < 1
    ):
        raise ValueError(f"{context}.source_file_count must be a positive integer")


def _validate_checkpoint_provenance(
    value: object,
    config: Mapping[str, object],
    *,
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} has invalid checkpoint provenance")
    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"{context}.checkpoint must be a non-empty path")
    _require_sha256(value.get("checkpoint_sha256"), context=f"{context}.checkpoint_sha256")
    step = value.get("checkpoint_step")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or step != config.get("checkpoint_step")
    ):
        raise ValueError(f"{context}.checkpoint_step does not match the job config")


def _require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_timestamp(value: object, *, context: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{context} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{context} must include a timezone")
    return parsed


def _validate_lifecycle(
    payload: Mapping[str, object],
    *,
    context: str,
    completion_fields: Sequence[str],
) -> None:
    """Validate the temporal and field invariants of a persisted state object."""
    status = payload.get("status")
    if status not in {"started", "complete"}:
        raise ValueError(f"{context} has an invalid status")
    started_at = _require_timestamp(payload.get("started_at"), context=f"{context} started_at")
    if status == "started":
        unexpected = sorted(field for field in completion_fields if field in payload)
        if unexpected:
            raise ValueError(f"{context} has completion fields while started: {unexpected}")
        return

    finished_at = _require_timestamp(payload.get("finished_at"), context=f"{context} finished_at")
    if finished_at < started_at:
        raise ValueError(f"{context} finished_at precedes started_at")


def _validate_nested_lifecycle(
    suite: Mapping[str, object],
    job: Mapping[str, object],
    *,
    context: str,
) -> None:
    suite_started = _require_timestamp(
        suite.get("started_at"), context=f"{context} suite started_at"
    )
    job_started = _require_timestamp(job.get("started_at"), context=f"{context} job started_at")
    job_finished = _require_timestamp(job.get("finished_at"), context=f"{context} job finished_at")
    if job_started < suite_started:
        raise ValueError(f"{context} job started before its suite")
    if suite.get("status") == "complete":
        suite_finished = _require_timestamp(
            suite.get("finished_at"), context=f"{context} suite finished_at"
        )
        if job_finished > suite_finished:
            raise ValueError(f"{context} job finished after its suite")


def _require_mapping(value: object, *, context: str) -> MutableMapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _require_list(value: object, *, context: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return value


def _record_key(record: Mapping[str, object]) -> tuple[int, str]:
    seed = record.get("seed")
    method = record.get("method")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Result record has an invalid seed: {seed!r}")
    if not isinstance(method, str) or not method:
        raise ValueError(f"Result record has an invalid method: {method!r}")
    return seed, method


def _finite_number(value: object, *, context: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return result


def _positive_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _non_negative_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _numeric_pair(value: object, *, context: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{context} must contain two finite numbers")
    return (
        _finite_number(value[0], context=f"{context}[0]"),
        _finite_number(value[1], context=f"{context}[1]"),
    )


def _validate_job_config(config: Mapping[str, object], *, context: str) -> None:
    """Reject persisted job configs that cannot execute the paper protocol."""
    for key in ("name", "model", "model_family", "dataset", "checkpoint", "data_dir"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{context}.{key} must be a non-empty string")

    task_type = config.get("task_type")
    if not isinstance(task_type, str) or not task_type:
        raise ValueError(f"{context}.task_type must be a non-empty string")
    metric = config.get("metric")
    metrics_by_task = {
        "lm": {"loss", "perplexity"},
        "cls": {"accuracy"},
        "reg": {"loss", "pearson"},
        "cv": {"accuracy"},
    }
    if task_type not in metrics_by_task or metric not in metrics_by_task[task_type]:
        raise ValueError(f"{context}.metric is incompatible with task_type")
    try:
        expected_family = get_model_type(str(config["model"]))
        expected_task_type = get_task_type(str(config["dataset"]))
    except ValueError as exc:
        raise ValueError(f"{context} names an unsupported model or dataset") from exc
    if config.get("model_family") != expected_family:
        raise ValueError(f"{context}.model_family does not match model")
    if task_type != expected_task_type:
        raise ValueError(f"{context}.task_type does not match dataset")

    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(
            isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(f"{context}.seeds must be unique non-negative integers")

    recovery_count = _positive_int(
        config.get("recovery_count"), context=f"{context}.recovery_count"
    )
    total_steps = _positive_int(config.get("total_steps"), context=f"{context}.total_steps")
    hvp_batches = _positive_int(config.get("hvp_batches"), context=f"{context}.hvp_batches")
    eval_batches = _positive_int(config.get("eval_batches"), context=f"{context}.eval_batches")
    train_pool_batches = _positive_int(
        config.get("train_pool_batches"), context=f"{context}.train_pool_batches"
    )
    _positive_int(config.get("checkpoint_step"), context=f"{context}.checkpoint_step")
    _positive_int(config.get("batch_size"), context=f"{context}.batch_size")
    _positive_int(config.get("seq_length"), context=f"{context}.seq_length")
    prune_ratio = _finite_number(config.get("prune_ratio"), context=f"{context}.prune_ratio")
    max_layer_ratio = _finite_number(
        config.get("max_layer_ratio"), context=f"{context}.max_layer_ratio"
    )
    if not 0.0 < prune_ratio < 1.0:
        raise ValueError(f"{context}.prune_ratio must be in (0, 1)")
    if not prune_ratio <= max_layer_ratio <= 1.0:
        raise ValueError(f"{context}.max_layer_ratio must cover prune_ratio and be <= 1")
    declared_ratios = config.get("prune_ratios")
    if (
        not isinstance(declared_ratios, list)
        or not declared_ratios
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) < 1.0
            for value in declared_ratios
        )
        or len(set(declared_ratios)) != len(declared_ratios)
        or prune_ratio not in declared_ratios
    ):
        raise ValueError(f"{context}.prune_ratio must belong to valid declared prune_ratios")
    declared_recoveries = config.get("recovery_counts")
    if (
        not isinstance(declared_recoveries, list)
        or not declared_recoveries
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in declared_recoveries
        )
        or len(set(declared_recoveries)) != len(declared_recoveries)
        or recovery_count not in declared_recoveries
    ):
        raise ValueError(f"{context}.recovery_count must belong to valid declared recovery_counts")
    max_recovery_count = max(declared_recoveries)
    if total_steps < max_recovery_count + 1:
        raise ValueError(f"{context}.total_steps must cover every declared recovery segment")
    required_pool = total_steps + max_recovery_count * hvp_batches
    if train_pool_batches < required_pool:
        raise ValueError(f"{context}.train_pool_batches must be at least {required_pool}")
    if max(declared_ratios) > max_layer_ratio:
        raise ValueError(f"{context}.max_layer_ratio must cover every declared prune ratio")
    learning_rate = _finite_number(config.get("learning_rate"), context=f"{context}.learning_rate")
    if learning_rate <= 0.0:
        raise ValueError(f"{context}.learning_rate must be positive")

    protocol = config.get("recovery_protocol")
    scheduler = protocol.get("scheduler") if isinstance(protocol, Mapping) else None
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("optimizer_state") != "restored_from_checkpoint"
        or protocol.get("learning_rate_source") != "manifest"
        or protocol.get("learning_rate") != learning_rate
        or not isinstance(scheduler, Mapping)
        or scheduler.get("name") != "cosine_annealing"
        or scheduler.get("state") != "fresh"
        or scheduler.get("scope") != "matched_recovery_horizon"
        or scheduler.get("t_max") != total_steps
        or scheduler.get("eta_min") != 0.0
    ):
        raise ValueError(f"{context}.recovery_protocol is incompatible with the job")


def _validate_training_evidence(
    value: object,
    *,
    expected_steps: int,
    task_type: str,
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    steps = value.get("steps")
    if isinstance(steps, bool) or steps != expected_steps:
        raise ValueError(f"{context} has incompatible steps")
    if value.get("task_type") != task_type:
        raise ValueError(f"{context} has incompatible task_type")
    _finite_number(value.get("seconds"), context=f"{context}.seconds", minimum=0.0)
    for key, minimum in (("train_losses", None), ("learning_rates", 0.0)):
        values = value.get(key)
        if not isinstance(values, list) or len(values) != expected_steps:
            raise ValueError(f"{context}.{key} must contain one value per step")
        for index, item in enumerate(values):
            _finite_number(
                item,
                context=f"{context}.{key}[{index}]",
                minimum=minimum,
            )


def _validate_recovery_lr_trace(
    cycles: Sequence[Mapping[str, object]],
    final_training: object,
    *,
    config: Mapping[str, object],
    context: str,
) -> None:
    protocol = config.get("recovery_protocol")
    scheduler = protocol.get("scheduler") if isinstance(protocol, Mapping) else None
    total_steps = config.get("total_steps")
    learning_rate = config.get("learning_rate")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("optimizer_state") != "restored_from_checkpoint"
        or protocol.get("learning_rate_source") != "manifest"
        or protocol.get("learning_rate") != learning_rate
        or not isinstance(scheduler, Mapping)
        or scheduler.get("name") != "cosine_annealing"
        or scheduler.get("state") != "fresh"
        or scheduler.get("scope") != "matched_recovery_horizon"
        or scheduler.get("t_max") != total_steps
        or scheduler.get("eta_min") != 0.0
    ):
        raise ValueError(f"{context} has an invalid recovery protocol")
    base_lr = _finite_number(learning_rate, context=f"{context}.learning_rate", minimum=0.0)
    if base_lr <= 0.0 or isinstance(total_steps, bool) or not isinstance(total_steps, int):
        raise ValueError(f"{context} has an invalid recovery learning rate")

    observed = []
    for training in [*(cycle["training"] for cycle in cycles), final_training]:
        if not isinstance(training, Mapping):
            raise ValueError(f"{context} has invalid training evidence")
        observed.extend(training["learning_rates"])
    expected = [
        base_lr * (1.0 + math.cos(math.pi * step / total_steps)) / 2.0
        for step in range(total_steps)
    ]
    if len(observed) != len(expected) or any(
        not math.isclose(float(actual), target, rel_tol=1e-9, abs_tol=1e-15)
        for actual, target in zip(observed, expected)
    ):
        raise ValueError(f"{context} learning rates do not match the declared cosine schedule")


def _validate_evaluation_evidence(
    value: object,
    *,
    metric: str,
    expected_batches: object,
    context: str,
) -> None:
    if not isinstance(value, Mapping) or metric not in value:
        raise ValueError(f"{context} is missing metric {metric}")
    numeric = _finite_number(value[metric], context=f"{context}.{metric}")
    if metric == "accuracy" and not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{context}.{metric} must be in [0, 1]")
    if metric == "pearson" and not -1.0 <= numeric <= 1.0:
        raise ValueError(f"{context}.{metric} must be in [-1, 1]")
    if metric == "perplexity" and numeric <= 0.0:
        raise ValueError(f"{context}.{metric} must be positive")
    if metric == "loss" and numeric < 0.0:
        raise ValueError(f"{context}.{metric} must be non-negative")
    _finite_number(value.get("seconds"), context=f"{context}.seconds", minimum=0.0)
    if (
        isinstance(expected_batches, bool)
        or not isinstance(expected_batches, int)
        or value.get("batches") != expected_batches
    ):
        raise ValueError(f"{context} does not match the declared evaluation batch count")
    examples = value.get("examples")
    if isinstance(examples, bool) or not isinstance(examples, int) or examples < expected_batches:
        raise ValueError(f"{context} has an invalid evaluated example count")


def _validate_compression_evidence(
    value: object,
    *,
    method: str,
    internal_method: object,
    prune_ratio: float,
    expected_config: Mapping[str, object],
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} is missing compression evidence")
    allocation = value.get("allocation")
    if not isinstance(allocation, Mapping):
        raise ValueError(f"{context} is missing allocation evidence")
    mask = value.get("mask")
    if not isinstance(mask, Mapping):
        raise ValueError(f"{context} is missing mask evidence")
    eligible = mask.get("eligible_parameters")
    pruned = mask.get("pruned")
    if isinstance(eligible, bool) or not isinstance(eligible, int) or eligible < 1:
        raise ValueError(f"{context} has invalid eligible parameter count")
    if isinstance(pruned, bool) or not isinstance(pruned, int):
        raise ValueError(f"{context} has invalid pruned parameter count")
    expected_pruned = math.floor(prune_ratio * eligible)
    if pruned != expected_pruned:
        raise ValueError(f"{context} pruned {pruned} parameters; expected {expected_pruned}")
    target_pruned = allocation.get("target_pruned")
    if (
        isinstance(target_pruned, bool)
        or not isinstance(target_pruned, int)
        or target_pruned != expected_pruned
    ):
        raise ValueError(
            f"{context} allocation target_pruned does not match the exact prune budget"
        )
    _finite_number(
        mask.get("residual_magnitude_cost"),
        context=f"{context}.mask.residual_magnitude_cost",
        minimum=0.0,
    )
    rates = mask.get("layer_rates")
    if not isinstance(rates, list) or not rates:
        raise ValueError(f"{context} has invalid layer rates")
    layer_sizes = allocation.get("layer_sizes")
    if (
        not isinstance(layer_sizes, list)
        or len(layer_sizes) != len(rates)
        or not all(
            isinstance(size, int) and not isinstance(size, bool) and size > 0
            for size in layer_sizes
        )
        or sum(layer_sizes) != eligible
    ):
        raise ValueError(f"{context} allocation layer sizes do not match the mask")
    layer_prune_counts = []
    for index, rate in enumerate(rates):
        numeric = _finite_number(rate, context=f"{context}.mask.layer_rates[{index}]")
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{context} layer rate must be in [0, 1]")
        count = round(numeric * layer_sizes[index])
        if not math.isclose(
            numeric,
            count / layer_sizes[index],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{context} layer rate is not an exact layer prune count")
        layer_prune_counts.append(count)
    if sum(layer_prune_counts) != pruned:
        raise ValueError(f"{context} layer rates do not add up to the pruned count")
    _validate_canonical_compression_metadata(
        value,
        allocation,
        internal_method=internal_method,
        expected_pruned=expected_pruned,
        eligible=eligible,
        layer_sizes=layer_sizes,
        context=context,
    )
    _validate_method_evidence(
        value,
        method=method,
        internal_method=internal_method,
        expected_config=expected_config,
        layer_count=len(layer_sizes),
        layer_prune_counts=layer_prune_counts,
        context=context,
    )


def _validate_scoring_metadata(
    value: object,
    *,
    score_kind: str,
    batch_key: str,
    expected_config: Mapping[str, object],
    layer_count: int,
    context: str,
) -> None:
    if not isinstance(value, Mapping) or value.get("score_kind") != score_kind:
        raise ValueError(f"{context} has incompatible scoring evidence")
    if value.get(batch_key) != expected_config.get("hvp_batches"):
        raise ValueError(f"{context} has incompatible scoring batch count")
    for key in ("task_type", "model_family"):
        if value.get(key) != expected_config.get(key):
            raise ValueError(f"{context} scoring {key} does not match the job")
    _finite_number(
        value.get("total_seconds"), context=f"{context}.scoring.total_seconds", minimum=0.0
    )
    layer_seconds = value.get("layer_seconds")
    if not isinstance(layer_seconds, list) or len(layer_seconds) != layer_count:
        raise ValueError(f"{context} has invalid per-layer scoring timings")
    for index, seconds in enumerate(layer_seconds):
        _finite_number(
            seconds,
            context=f"{context}.scoring.layer_seconds[{index}]",
            minimum=0.0,
        )
    before = _numeric_pair(
        value.get("checksum_before"), context=f"{context}.scoring.checksum_before"
    )
    after = _numeric_pair(value.get("checksum_after"), context=f"{context}.scoring.checksum_after")
    delta = _numeric_pair(value.get("checksum_delta"), context=f"{context}.scoring.checksum_delta")
    if before != after or any(item != 0.0 for item in delta):
        raise ValueError(f"{context} scoring changed model parameters")
    if value.get("changed_parameter_versions") != []:
        raise ValueError(f"{context} scoring reports changed parameter versions")
    if value.get("optimizer_constructed") is not False or value.get("model_mode") != "eval":
        raise ValueError(f"{context} has incompatible scoring execution metadata")


def _validate_canonical_compression_metadata(
    value: Mapping[str, object],
    allocation: Mapping[str, object],
    *,
    internal_method: object,
    expected_pruned: int,
    eligible: int,
    layer_sizes: Sequence[int],
    context: str,
) -> None:
    """Validate the compact score/allocation summary on new paper records."""
    expected = _CANONICAL_COMPRESSION_METHODS.get(internal_method)
    if expected is None:
        return
    required = (
        "score_kind",
        "allocation_kind",
        "target_pruned",
        "layer_sizes",
    )
    present = [key in value for key in required]
    if internal_method in _REQUIRED_CANONICAL_COMPRESSION_METHODS and not all(present):
        raise ValueError(f"{context} is missing canonical score/allocation metadata")
    if not any(present):
        return
    if not all(present):
        raise ValueError(f"{context} has incomplete canonical score/allocation metadata")
    expected_score_kind, expected_allocation_kind = expected
    if value.get("score_kind") != expected_score_kind:
        raise ValueError(f"{context} has incompatible score_kind")
    if value.get("allocation_kind") != expected_allocation_kind:
        raise ValueError(f"{context} has incompatible allocation_kind")
    if value.get("target_pruned") != expected_pruned:
        raise ValueError(f"{context} target_pruned does not match the exact prune budget")
    if value.get("layer_sizes") != list(layer_sizes):
        raise ValueError(f"{context} layer sizes do not match the mask")
    if allocation.get("target_pruned") != expected_pruned:
        raise ValueError(f"{context} allocation target_pruned does not match the summary")
    if allocation.get("layer_sizes") != list(layer_sizes):
        raise ValueError(f"{context} allocation layer sizes do not match the summary")
    nested_kind = allocation.get("allocation_kind")
    if nested_kind is not None and nested_kind != expected_allocation_kind:
        raise ValueError(f"{context} allocation kind does not match the summary")
    if value.get("score_kind") == "taylor_hvp":
        scoring = value.get("scoring")
        if isinstance(scoring, Mapping) and scoring.get("score_kind") != "taylor_hvp":
            raise ValueError(f"{context} scoring evidence does not match score_kind")


def _validate_method_evidence(
    compression: Mapping[str, object],
    *,
    method: str,
    internal_method: object,
    expected_config: Mapping[str, object],
    layer_count: int,
    layer_prune_counts: Sequence[int],
    context: str,
) -> None:
    allocation = compression["allocation"]
    if not isinstance(allocation, Mapping):  # Guarded by the caller.
        raise ValueError(f"{context} is missing allocation evidence")
    if method == "excp_style":
        if (
            compression.get("score_kind") != "residual_magnitude"
            or compression.get("scoring_batches") != 0
            or allocation.get("allocation") != "exact_global"
        ):
            raise ValueError(f"{context} has incompatible ExCP-style evidence")
        return
    if method == "inshrinkerator_style":
        if (
            compression.get("score_probe") != "current_full_weight"
            or compression.get("application_scope") != "matched_checkpoint_residual"
            or allocation.get("allocation") != "exact_global"
        ):
            raise ValueError(f"{context} has incompatible Inshrinkerator-style evidence")
        _validate_scoring_metadata(
            compression.get("scoring"),
            score_kind="first_order",
            batch_key="scoring_batches",
            expected_config=expected_config,
            layer_count=layer_count,
            context=context,
        )
        return
    if internal_method in {
        RESIDUAL_MAGNITUDE_UNIFORM_METHOD,
        TAYLOR_UNIFORM_METHOD,
        TAYLOR_SIGNED_UNIFORM_METHOD,
        TAYLOR_MEANABS_UNIFORM_METHOD,
    }:
        uniform_score_kinds = {
            RESIDUAL_MAGNITUDE_UNIFORM_METHOD: "residual_magnitude",
            TAYLOR_UNIFORM_METHOD: "taylor_hvp",
            TAYLOR_SIGNED_UNIFORM_METHOD: "taylor_hvp_signed",
            TAYLOR_SIGNED_FIRST_ORDER_UNIFORM_METHOD: "taylor_hvp_signed_fo",
            TAYLOR_SIGNED_SECOND_ORDER_UNIFORM_METHOD: "taylor_hvp_signed_so",
            TAYLOR_MEANABS_UNIFORM_METHOD: "taylor_hvp_mean_abs",
        }
        expected_score_kind = uniform_score_kinds[internal_method]
        if (
            compression.get("score_kind") != expected_score_kind
            or compression.get("allocation_kind") != "uniform_per_layer"
            or allocation.get("allocation") != "uniform_per_layer"
            or allocation.get("uniform_layer_counts") != list(layer_prune_counts)
            or allocation.get("target_eligible_sparsity") != expected_config.get("prune_ratio")
        ):
            raise ValueError(f"{context} has incompatible uniform allocation evidence")
        if expected_score_kind == "residual_magnitude":
            if compression.get("scoring_batches") != 0:
                raise ValueError(f"{context} has incompatible magnitude scoring evidence")
        else:
            _validate_scoring_metadata(
                compression.get("scoring"),
                score_kind=expected_score_kind,
                batch_key="hvp_batches",
                expected_config=expected_config,
                layer_count=layer_count,
                context=context,
            )
        return
    if internal_method in {
        RESIDUAL_MAGNITUDE_WEIBULL_MOM_METHOD,
        TAYLOR_WEIBULL_MOM_METHOD,
        TAYLOR_SIGNED_WEIBULL_MOM_METHOD,
        TAYLOR_MEANABS_WEIBULL_MOM_METHOD,
    }:
        weibull_score_kinds = {
            RESIDUAL_MAGNITUDE_WEIBULL_MOM_METHOD: "residual_magnitude",
            TAYLOR_WEIBULL_MOM_METHOD: "taylor_hvp",
            TAYLOR_SIGNED_WEIBULL_MOM_METHOD: "taylor_hvp_signed",
            TAYLOR_MEANABS_WEIBULL_MOM_METHOD: "taylor_hvp_mean_abs",
        }
        expected_weibull_score_kind = weibull_score_kinds[internal_method]
        fits = allocation.get("weibull_fits")
        counts = allocation.get("weibull_layer_counts")
        if (
            not isinstance(fits, list)
            or len(fits) != layer_count
            or not all(
                isinstance(fit, Mapping)
                and fit.get("layer") == index
                and isinstance(fit.get("valid"), bool)
                for index, fit in enumerate(fits)
            )
            or not isinstance(counts, list)
            or len(counts) != layer_count
            or not all(
                isinstance(count, int) and not isinstance(count, bool) and count >= 0
                for count in counts
            )
            or counts != list(layer_prune_counts)
            or sum(counts) != allocation.get("target_pruned")
            or allocation.get("eligible_parameters") != sum(allocation["layer_sizes"])
            or allocation.get("target_eligible_sparsity") != expected_config.get("prune_ratio")
            or not isinstance(allocation.get("weibull"), Mapping)
        ):
            raise ValueError(f"{context} is missing DACP Weibull allocation evidence")
        _validate_weibull_evidence(
            fits,
            counts,
            layer_sizes=allocation["layer_sizes"],
            target_pruned=allocation["target_pruned"],
            prune_ratio=expected_config.get("prune_ratio"),
            max_layer_ratio=expected_config.get("max_layer_ratio"),
            metadata=allocation["weibull"],
            context=context,
        )
        if internal_method == RESIDUAL_MAGNITUDE_WEIBULL_MOM_METHOD:
            if (
                compression.get("score_kind") != "residual_magnitude"
                or compression.get("scoring_batches") != 0
            ):
                raise ValueError(f"{context} has incompatible magnitude scoring evidence")
        else:
            _validate_scoring_metadata(
                compression.get("scoring"),
                score_kind=expected_weibull_score_kind,
                batch_key="hvp_batches",
                expected_config=expected_config,
                layer_count=layer_count,
                context=context,
            )
        return
    if internal_method in {TAYLOR_EXACT_GLOBAL_METHOD, TAYLOR_SIGNED_EXACT_GLOBAL_METHOD}:
        expected_score_kind = (
            "taylor_hvp_signed"
            if internal_method == TAYLOR_SIGNED_EXACT_GLOBAL_METHOD
            else "taylor_hvp"
        )
        if (
            compression.get("score_kind") != expected_score_kind
            or compression.get("allocation_kind") != "exact_global"
            or allocation.get("allocation") != "exact_global"
        ):
            raise ValueError(f"{context} has incompatible Taylor exact-global evidence")
        _validate_scoring_metadata(
            compression.get("scoring"),
            score_kind=expected_score_kind,
            batch_key="hvp_batches",
            expected_config=expected_config,
            layer_count=layer_count,
            context=context,
        )
        return
    raise ValueError(f"{context} uses unsupported compressed method {method!r}")


def _validate_weibull_evidence(
    fits: Sequence[Mapping[str, object]],
    counts: Sequence[int],
    *,
    layer_sizes: object,
    target_pruned: object,
    prune_ratio: object,
    max_layer_ratio: object,
    metadata: object,
    context: str,
) -> None:
    if not isinstance(layer_sizes, list) or not isinstance(metadata, Mapping):
        raise ValueError(f"{context} has malformed DACP Weibull evidence")
    target = _non_negative_int(target_pruned, context=f"{context}.allocation.target_pruned")
    ratio = _finite_number(prune_ratio, context=f"{context}.config.prune_ratio")
    layer_cap = _finite_number(max_layer_ratio, context=f"{context}.config.max_layer_ratio")
    capacities = [math.floor(layer_cap * size) for size in layer_sizes]
    if any(count > capacity for count, capacity in zip(counts, capacities)):
        raise ValueError(f"{context} exceeds the declared per-layer prune cap")

    valid_count = 0
    for index, (fit, size) in enumerate(zip(fits, layer_sizes)):
        if fit.get("count") != size:
            raise ValueError(f"{context} Weibull fit count does not match its layer")
        zero_fraction = _finite_number(
            fit.get("zero_fraction"),
            context=f"{context}.allocation.weibull_fits[{index}].zero_fraction",
        )
        if not 0.0 <= zero_fraction <= 1.0:
            raise ValueError(f"{context} has invalid Weibull zero_fraction")
        maximum = _finite_number(
            fit.get("maximum"),
            context=f"{context}.allocation.weibull_fits[{index}].maximum",
            minimum=0.0,
        )
        mean = _finite_number(
            fit.get("mean"),
            context=f"{context}.allocation.weibull_fits[{index}].mean",
            minimum=0.0,
        )
        variance = _finite_number(
            fit.get("variance"),
            context=f"{context}.allocation.weibull_fits[{index}].variance",
            minimum=0.0,
        )
        cv_squared = _finite_number(
            fit.get("cv_squared"),
            context=f"{context}.allocation.weibull_fits[{index}].cv_squared",
            minimum=0.0,
        )
        expected_cv = variance / (mean * mean) if mean > 0.0 else 0.0
        if not math.isclose(cv_squared, expected_cv, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{context} Weibull fit moments are internally inconsistent")
        reduction = fit.get("moment_reduction")
        if not isinstance(reduction, Mapping) or reduction != {
            "distributed": False,
            "world_size": 1,
            "backend": None,
            "communicated_scalars_per_rank": 0,
        }:
            raise ValueError(f"{context} has incompatible Weibull moment reduction evidence")
        if fit["valid"]:
            valid_count += 1
            numeric_fit: dict[str, float] = {}
            for key, minimum in (
                ("mean", 0.0),
                ("variance", 0.0),
                ("cv_squared", 0.0),
                ("shape", 0.0),
                ("scale", 0.0),
            ):
                value = _finite_number(
                    fit.get(key),
                    context=f"{context}.allocation.weibull_fits[{index}].{key}",
                    minimum=minimum,
                )
                if key in {"mean", "cv_squared", "shape", "scale"} and value <= 0.0:
                    raise ValueError(f"{context} has non-positive valid Weibull {key}")
                numeric_fit[key] = value
            shape = numeric_fit["shape"]
            scale = numeric_fit["scale"]
            distribution_cv = (
                math.exp(math.lgamma(1.0 + 2.0 / shape) - 2.0 * math.lgamma(1.0 + 1.0 / shape))
                - 1.0
            )
            expected_scale = mean / math.exp(math.lgamma(1.0 + 1.0 / shape))
            if (
                not math.isclose(cv_squared, expected_cv, rel_tol=1e-9, abs_tol=1e-12)
                or not math.isclose(cv_squared, distribution_cv, rel_tol=1e-7, abs_tol=1e-10)
                or not math.isclose(scale, expected_scale, rel_tol=1e-7, abs_tol=1e-10)
            ):
                raise ValueError(f"{context} Weibull fit moments are internally inconsistent")
        else:
            reason = fit.get("reason")
            if reason not in _WEIBULL_INVALID_REASONS - {"empty values"}:
                raise ValueError(f"{context} invalid Weibull fit has an unsupported reason")
            if reason == "non-positive maximum" and maximum > 0.0:
                raise ValueError(f"{context} invalid Weibull reason contradicts its maximum")
            if reason == "degenerate moments" and not (mean <= 1e-15 or variance <= 0.0):
                raise ValueError(f"{context} invalid Weibull reason contradicts its moments")
            if reason == "degenerate coefficient of variation" and not (
                cv_squared < 1e-10 or not math.isfinite(cv_squared)
            ):
                raise ValueError(f"{context} invalid Weibull reason contradicts its moments")

    fallback = metadata.get("fallback")
    from experiments.lib.residual_budget import largest_remainder_counts

    if valid_count:
        threshold = _finite_number(
            metadata.get("threshold"),
            context=f"{context}.allocation.weibull.threshold",
            minimum=0.0,
        )
        real_counts = metadata.get("real_counts")
        if (
            fallback is not None
            or metadata.get("capacities") != capacities
            or not isinstance(real_counts, list)
            or len(real_counts) != len(layer_sizes)
        ):
            raise ValueError(f"{context} has invalid Weibull threshold allocation evidence")
        normalized_real = [
            _finite_number(
                value,
                context=f"{context}.allocation.weibull.real_counts[{index}]",
                minimum=0.0,
            )
            for index, value in enumerate(real_counts)
        ]
        if any(value > capacity for value, capacity in zip(normalized_real, capacities)):
            raise ValueError(f"{context} Weibull real counts exceed layer capacity")
        recomputed_real = []
        for index, (fit, size, capacity) in enumerate(zip(fits, layer_sizes, capacities)):
            if fit["valid"]:
                shape = _finite_number(
                    fit.get("shape"),
                    context=f"{context}.allocation.weibull_fits[{index}].shape",
                )
                scale = _finite_number(
                    fit.get("scale"),
                    context=f"{context}.allocation.weibull_fits[{index}].scale",
                )
                if threshold <= 0.0:
                    cdf = 0.0
                else:
                    log_power = shape * (math.log(threshold) - math.log(scale))
                    if log_power > 40:
                        cdf = 1.0
                    elif log_power < -40:
                        cdf = math.exp(log_power)
                    else:
                        cdf = -math.expm1(-math.exp(log_power))
                real_count = size * cdf
            else:
                real_count = size * ratio
            recomputed_real.append(min(real_count, float(capacity)))
        if any(
            not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10)
            for actual, expected in zip(normalized_real, recomputed_real)
        ):
            raise ValueError(f"{context} Weibull real counts do not match the fitted threshold")
        minimum_total = sum(
            min(size * ratio, float(capacity))
            for fit, size, capacity in zip(fits, layer_sizes, capacities)
            if not fit["valid"]
        )
        if threshold == 0.0:
            if minimum_total < target:
                raise ValueError(f"{context} has a zero Weibull threshold below the target")
        elif not math.isclose(
            sum(recomputed_real),
            float(target),
            rel_tol=1e-8,
            abs_tol=1e-6,
        ):
            raise ValueError(f"{context} Weibull threshold does not solve the target budget")
        expected_counts = largest_remainder_counts(normalized_real, target, capacities)
    else:
        if (
            metadata.get("threshold") is not None
            or fallback != "all Weibull fits invalid"
            or "real_counts" in metadata
            or "capacities" in metadata
        ):
            raise ValueError(f"{context} has invalid Weibull fallback evidence")
        expected_counts = largest_remainder_counts(
            [ratio * size for size in layer_sizes],
            target,
            capacities,
        )
    if list(counts) != expected_counts:
        raise ValueError(f"{context} Weibull counts do not match allocation evidence")


def _validate_complete_record(
    record: Mapping[str, object],
    *,
    expected_config: Mapping[str, object],
    method_contracts: Mapping[str, Mapping[str, object]],
    metric: str,
) -> tuple[int, str]:
    seed, method = _record_key(record)
    declared_seeds = expected_config.get("seeds")
    if not isinstance(declared_seeds, list) or seed not in declared_seeds:
        raise ValueError(f"Result record uses undeclared seed {seed}")
    contract = method_contracts.get(method)
    if contract is None:
        raise ValueError(f"Result record uses undeclared method {method!r}")
    for key in ("owner", "fidelity", "internal_method"):
        if record.get(key) != contract.get(key):
            raise ValueError(f"Result record {seed}/{method} has incompatible {key}")
    for key in ("prune_ratio", "recovery_count"):
        if record.get(key) != expected_config.get(key):
            raise ValueError(f"Result record {seed}/{method} has incompatible {key}")

    recovery_count = expected_config.get("recovery_count")
    total_steps = expected_config.get("total_steps")
    task_type = expected_config.get("task_type")
    cycles = record.get("cycles")
    if (
        isinstance(recovery_count, bool)
        or not isinstance(recovery_count, int)
        or isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or not isinstance(task_type, str)
        or not task_type
        or not isinstance(cycles, list)
    ):
        raise ValueError(f"Result record {seed}/{method} has invalid recovery cycles")
    if len(cycles) != recovery_count:
        raise ValueError(f"Result record {seed}/{method} is incomplete")
    base_length, extra = divmod(total_steps, recovery_count + 1)
    segment_lengths = [
        base_length + (1 if index < extra else 0) for index in range(recovery_count + 1)
    ]
    for index, cycle in enumerate(cycles, start=1):
        if not isinstance(cycle, Mapping) or cycle.get("cycle") != index:
            raise ValueError(f"Result record {seed}/{method} has invalid cycle {index}")
        required = {"training", "before_recovery", "after_recovery", "compression"}
        if not required.issubset(cycle):
            raise ValueError(f"Result record {seed}/{method} is missing cycle fields")
        _validate_training_evidence(
            cycle["training"],
            expected_steps=segment_lengths[index - 1],
            task_type=task_type,
            context=f"Result record {seed}/{method} cycle {index} training",
        )
        for stage in ("before_recovery", "after_recovery"):
            _validate_evaluation_evidence(
                cycle[stage],
                metric=metric,
                expected_batches=expected_config.get("eval_batches"),
                context=f"Result record {seed}/{method} {stage}",
            )
        compression = cycle["compression"]
        if method == "no_compression":
            if compression is not None:
                raise ValueError(f"Control record {seed}/{method} must not compress")
            if cycle["after_recovery"] != cycle["before_recovery"]:
                raise ValueError(
                    f"Control record {seed}/{method} must preserve metrics across recovery"
                )
        else:
            prune_ratio = _finite_number(
                expected_config.get("prune_ratio"),
                context=f"Result record {seed}/{method} prune_ratio",
            )
            _validate_compression_evidence(
                compression,
                method=method,
                internal_method=contract.get("internal_method"),
                prune_ratio=prune_ratio,
                expected_config=expected_config,
                context=f"Compressed record {seed}/{method} cycle {index}",
            )

    _validate_training_evidence(
        record.get("final_training"),
        expected_steps=segment_lengths[-1],
        task_type=task_type,
        context=f"Result record {seed}/{method} final training",
    )
    _validate_recovery_lr_trace(
        cycles,
        record.get("final_training"),
        config=expected_config,
        context=f"Result record {seed}/{method}",
    )

    _validate_evaluation_evidence(
        record.get("final"),
        metric=metric,
        expected_batches=expected_config.get("eval_batches"),
        context=f"Result record {seed}/{method} final evaluation",
    )
    wall_seconds = record.get("wall_seconds")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(float(wall_seconds))
        or wall_seconds < 0
    ):
        raise ValueError(f"Result record {seed}/{method} has invalid wall_seconds")
    return seed, method


def _validate_batch_plans(
    plans: Mapping[str, object],
    config: Mapping[str, object],
    training_pool: object,
    *,
    path: Path,
) -> None:
    recovery_count = _positive_int(
        config.get("recovery_count"), context=f"Job state {path} recovery_count"
    )
    total_steps = _positive_int(config.get("total_steps"), context=f"Job state {path} total_steps")
    hvp_batches = _positive_int(config.get("hvp_batches"), context=f"Job state {path} hvp_batches")
    if not isinstance(training_pool, Mapping):
        raise ValueError(f"Job state {path} has invalid training pool identity")
    pool_count = training_pool.get("count")
    pool_digest = training_pool.get("sha256")
    if (
        isinstance(pool_count, bool)
        or not isinstance(pool_count, int)
        or pool_count < 1
        or not isinstance(pool_digest, str)
    ):
        raise ValueError(f"Job state {path} has invalid training pool identity")
    _require_sha256(pool_digest, context=f"Job state {path} training pool digest")
    if pool_count != config.get("train_pool_batches"):
        raise ValueError(f"Job state {path} training pool count does not match the config")

    expected_selected = total_steps + recovery_count * hvp_batches
    for seed, raw_plan in plans.items():
        if not isinstance(raw_plan, Mapping):
            raise ValueError(f"Job state {path} has malformed batch plan {seed}")
        indices = raw_plan.get("selected_pool_indices")
        training_hashes = raw_plan.get("training_segments")
        scoring_hashes = raw_plan.get("scoring_batches")
        if (
            not isinstance(indices, list)
            or len(indices) != expected_selected
            or not all(
                isinstance(index, int) and not isinstance(index, bool) and 0 <= index < pool_count
                for index in indices
            )
            or len(set(indices)) != len(indices)
        ):
            raise ValueError(f"Job state {path} has invalid selected indices for seed {seed}")
        if (
            not isinstance(training_hashes, list)
            or len(training_hashes) != recovery_count + 1
            or not all(
                isinstance(value, str) and _SHA256_PATTERN.fullmatch(value)
                for value in training_hashes
            )
            or not isinstance(scoring_hashes, list)
            or len(scoring_hashes) != recovery_count
            or not all(
                isinstance(value, str) and _SHA256_PATTERN.fullmatch(value)
                for value in scoring_hashes
            )
        ):
            raise ValueError(f"Job state {path} has invalid batch hashes for seed {seed}")


def _validate_evaluation_identity(
    value: object,
    config: Mapping[str, object],
    *,
    path: Path,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"Job state {path} has invalid evaluation batch identity")
    count = value.get("count")
    digest = value.get("sha256")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count != config.get("eval_batches")
        or not isinstance(digest, str)
    ):
        raise ValueError(f"Job state {path} has invalid evaluation batch identity")
    _require_sha256(digest, context=f"Job state {path} evaluation batch digest")


def _expected_aggregate(
    records: Sequence[Mapping[str, object]],
    metric: str,
) -> Mapping[str, object]:
    from experiments.lib.paper_runner import aggregate_job_results

    aggregate = aggregate_job_results(records, metric)
    if not isinstance(aggregate, Mapping):
        raise TypeError("Paper result aggregation must return a mapping")
    return aggregate
