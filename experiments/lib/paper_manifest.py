"""Loading and validation for the declaration-driven paper experiment suite."""

from __future__ import annotations

import os
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
    try:
        converted = tuple(cast(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} contains an invalid value") from exc
    if len(set(converted)) != len(converted):
        raise ValueError(f"{key} must not contain duplicates")
    return converted


def load_paper_manifest(path: Path) -> PaperManifest:
    """Load a paper suite and reject ambiguous or unsupported experiment claims."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Paper manifest must be a mapping")
    if payload.get("schema_version") != 1:
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
    for gate in gates.values():
        if not isinstance(gate, Mapping):
            raise ValueError("Every claim gate must be a mapping")
        if gate.get("enabled", True):
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
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"Workload names must be unique non-empty strings: {name!r}")
        names.add(name)
        model = str(record.get("model"))
        dataset = str(record.get("dataset"))
        family = get_model_type(model)
        task_type = get_task_type(dataset)
        seeds = _number_tuple(record, "seeds", int)
        ratios = _number_tuple(record, "prune_ratios", float)
        recoveries = _number_tuple(record, "recovery_counts", int)
        if any(seed < 0 for seed in seeds):
            raise ValueError(f"{name}: seeds must be non-negative")
        if any(not 0 < ratio < 1 for ratio in ratios):
            raise ValueError(f"{name}: prune_ratios must be in (0, 1)")
        max_layer_ratio = float(record.get("max_layer_ratio", 0.95))
        if not max(ratios) <= max_layer_ratio <= 1:
            raise ValueError(
                f"{name}: max_layer_ratio must cover every prune ratio and be <= 1"
            )
        if any(count < 1 for count in recoveries):
            raise ValueError(f"{name}: recovery_counts must be positive")
        total_steps = _positive_int(record, "total_steps")
        hvp_batches = _positive_int(record, "hvp_batches")
        required_pool = total_steps + max(recoveries) * hvp_batches
        train_pool_batches = int(record.get("train_pool_batches", required_pool))
        if train_pool_batches < required_pool:
            raise ValueError(
                f"{name}: train_pool_batches must be at least {required_pool}"
            )
        workloads.append(
            PaperWorkload(
                name=name,
                model=model,
                model_family=family,
                dataset=dataset,
                task_type=task_type,
                checkpoint=_expand_path(record.get("checkpoint"), path),
                data_dir=_expand_path(record.get("data_dir"), path),
                metric=str(record.get("metric")),
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
                learning_rate=float(record.get("learning_rate", 5e-5)),
            )
        )
    return PaperManifest(path, 1, methods, gates, tuple(workloads))


__all__ = ["PaperManifest", "PaperWorkload", "load_paper_manifest"]
