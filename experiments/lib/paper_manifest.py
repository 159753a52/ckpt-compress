"""Loading and validation for the declaration-driven paper experiment suite."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from experiments.lib.data import get_task_type
from experiments.lib.models import get_model_type
from experiments.lib.paper_baselines import (
    MethodContract,
    resolve_method_contracts,
    validate_claim_gate,
)


@dataclass(frozen=True)
class PaperWorkload:
    name: str
    model: str
    model_family: str
    dataset: str
    task_type: str
    checkpoint: Path
    checkpoint_step: int
    data_dir: Path
    metric: str
    seeds: tuple[int, ...]
    prune_ratios: tuple[float, ...]
    max_layer_ratio: float
    recovery_counts: tuple[int, ...]
    batch_size: int
    seq_length: int
    total_steps: int
    hvp_batches: int
    eval_batches: int
    train_pool_batches: int
    learning_rate: float


@dataclass(frozen=True)
class PaperManifest:
    path: Path
    schema_version: int
    methods: tuple[MethodContract, ...]
    claim_gates: Mapping[str, Mapping[str, object]]
    workloads: tuple[PaperWorkload, ...]


def _positive_int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer, got {value}")
    return value


def _expand_path(value: object, manifest_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest path must be a non-empty string, got {value!r}")
    expanded = Path(os.path.expandvars(value)).expanduser()
    if not expanded.is_absolute():
        expanded = manifest_path.parent.parent.parent / expanded
    return expanded.resolve()


def _number_tuple(record: Mapping[str, object], key: str, cast) -> tuple:
    values = record.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} must be a non-empty list")
    if cast is int:
        valid = all(isinstance(value, int) and not isinstance(value, bool) for value in values)
    else:
        valid = all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
        )
    if not valid:
        raise ValueError(f"{key} contains an invalid value")
    try:
        converted = tuple(cast(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} contains an invalid value") from exc
    if any(isinstance(value, float) and not math.isfinite(value) for value in converted):
        raise ValueError(f"{key} must contain only finite values")
    if len(set(converted)) != len(converted):
        raise ValueError(f"{key} must not contain duplicates")
    return converted


def _finite_float(record: Mapping[str, object], key: str, default: float) -> float:
    raw = record.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise ValueError(f"{key} must be a finite number, got {raw}")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be a finite number, got {raw}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number, got {raw}")
    return value


_METRICS_BY_TASK = {
    "lm": {"loss", "perplexity"},
    "cls": {"accuracy"},
    "reg": {"loss", "pearson"},
    "cv": {"accuracy"},
}
_WORKLOAD_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def paper_job_id(workload_name: object, prune_ratio: object, recovery_count: object) -> str:
    """Return the canonical, path-safe identity for one planned paper job."""
    if not isinstance(workload_name, str) or not _WORKLOAD_NAME_PATTERN.fullmatch(workload_name):
        raise ValueError(f"Invalid workload name for paper job: {workload_name!r}")
    if (
        isinstance(prune_ratio, bool)
        or not isinstance(prune_ratio, (int, float))
        or not math.isfinite(float(prune_ratio))
        or not 0.0 < float(prune_ratio) < 1.0
    ):
        raise ValueError(f"Invalid prune ratio for paper job: {prune_ratio!r}")
    if (
        isinstance(recovery_count, bool)
        or not isinstance(recovery_count, int)
        or recovery_count < 1
    ):
        raise ValueError(f"Invalid recovery count for paper job: {recovery_count!r}")
    ratio = float(prune_ratio).hex().replace("0x", "").replace(".", "p").replace("+", "")
    return f"{workload_name}_p{ratio}_k{recovery_count}"


def load_paper_manifest(path: Path) -> PaperManifest:
    """Load a paper suite and reject ambiguous or unsupported experiment claims."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Paper manifest must be a mapping")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("Paper manifest schema_version must be 1")

    method_names = payload.get("methods")
    if not isinstance(method_names, list) or not all(
        isinstance(name, str) for name in method_names
    ):
        raise ValueError("methods must be a list of method names")
    methods = tuple(resolve_method_contracts(method_names))
    gates = payload.get("claim_gates", {})
    if not isinstance(gates, Mapping):
        raise ValueError("claim_gates must be a mapping")
    for gate_name, gate in gates.items():
        if not isinstance(gate_name, str) or not gate_name:
            raise ValueError("Claim gate names must be non-empty strings")
        if not isinstance(gate, Mapping):
            raise ValueError("Every claim gate must be a mapping")
        enabled = gate.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"Claim gate {gate_name!r} enabled must be boolean")
        if enabled:
            validate_claim_gate(methods, gate)

    defaults = payload.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise ValueError("defaults must be a mapping")
    raw_workloads = payload.get("workloads")
    if not isinstance(raw_workloads, list) or not raw_workloads:
        raise ValueError("workloads must be a non-empty list")
    workloads = []
    names = set()
    for raw in raw_workloads:
        if not isinstance(raw, Mapping):
            raise ValueError("Every workload must be a mapping")
        record = {**defaults, **raw}
        name = record.get("name")
        if not isinstance(name, str) or not _WORKLOAD_NAME_PATTERN.fullmatch(name) or name in names:
            raise ValueError(
                "Workload names must be unique and contain only letters, digits, "
                f"underscores, or hyphens: {name!r}"
            )
        names.add(name)
        model = record.get("model")
        dataset = record.get("dataset")
        if not isinstance(model, str) or not model:
            raise ValueError(f"{name}: model must be a non-empty string")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"{name}: dataset must be a non-empty string")
        family = get_model_type(model)
        task_type = get_task_type(dataset)
        seeds = _number_tuple(record, "seeds", int)
        ratios = _number_tuple(record, "prune_ratios", float)
        recoveries = _number_tuple(record, "recovery_counts", int)
        if any(seed < 0 for seed in seeds):
            raise ValueError(f"{name}: seeds must be non-negative")
        if any(not 0 < ratio < 1 for ratio in ratios):
            raise ValueError(f"{name}: prune_ratios must be in (0, 1)")
        max_layer_ratio = _finite_float(record, "max_layer_ratio", 0.95)
        if not max(ratios) <= max_layer_ratio <= 1:
            raise ValueError(f"{name}: max_layer_ratio must cover every prune ratio and be <= 1")
        if any(count < 1 for count in recoveries):
            raise ValueError(f"{name}: recovery_counts must be positive")
        total_steps = _positive_int(record, "total_steps")
        hvp_batches = _positive_int(record, "hvp_batches")
        if total_steps < max(recoveries) + 1:
            raise ValueError(
                f"{name}: total_steps must give every one of the K+1 training segments a step"
            )
        required_pool = total_steps + max(recoveries) * hvp_batches
        train_pool_batches = _positive_int(
            {"train_pool_batches": record.get("train_pool_batches", required_pool)},
            "train_pool_batches",
        )
        if train_pool_batches < required_pool:
            raise ValueError(f"{name}: train_pool_batches must be at least {required_pool}")
        metric = record.get("metric")
        if not isinstance(metric, str) or metric not in _METRICS_BY_TASK[task_type]:
            raise ValueError(
                f"{name}: metric {metric!r} is unsupported for task type {task_type}; "
                f"expected one of {sorted(_METRICS_BY_TASK[task_type])}"
            )
        learning_rate = _finite_float(record, "learning_rate", 5e-5)
        if learning_rate <= 0:
            raise ValueError(f"{name}: learning_rate must be positive")
        workloads.append(
            PaperWorkload(
                name=name,
                model=model,
                model_family=family,
                dataset=dataset,
                task_type=task_type,
                checkpoint=_expand_path(record.get("checkpoint"), path),
                checkpoint_step=_positive_int(record, "checkpoint_step"),
                data_dir=_expand_path(record.get("data_dir"), path),
                metric=metric,
                seeds=seeds,
                prune_ratios=ratios,
                max_layer_ratio=max_layer_ratio,
                recovery_counts=recoveries,
                batch_size=_positive_int(record, "batch_size"),
                seq_length=_positive_int(record, "seq_length"),
                total_steps=total_steps,
                hvp_batches=hvp_batches,
                eval_batches=_positive_int(record, "eval_batches"),
                train_pool_batches=train_pool_batches,
                learning_rate=learning_rate,
            )
        )
    return PaperManifest(path, 1, methods, gates, tuple(workloads))


__all__ = ["PaperManifest", "PaperWorkload", "load_paper_manifest", "paper_job_id"]
