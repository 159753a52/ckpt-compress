"""Compatibility layer for experiment result JSON schemas."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml


@dataclass(frozen=True)
class ResultBundle:
    """Normalized records and configuration loaded from one result file."""

    records: List[Dict[str, Any]]
    config: Dict[str, Any]
    schema: str
    source_path: Optional[Path] = None


def _normalize_records(value: Any, schema: str) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{schema} results must be a list")

    records = []
    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            raise ValueError(f"{schema} result at index {index} must be an object")
        records.append(dict(record))
    return records


def _normalize_config(value: Any, schema: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{schema} config must be an object")
    return dict(value)


def normalize_result_payload(payload: Any) -> ResultBundle:
    """Normalize all result schemas written by current and legacy experiments."""
    if isinstance(payload, list):
        return ResultBundle(_normalize_records(payload, "list"), {}, "list")
    if not isinstance(payload, Mapping):
        raise ValueError("result payload must be a list or object")

    if "results" in payload:
        return ResultBundle(
            _normalize_records(payload["results"], "results"),
            _normalize_config(payload.get("config"), "results"),
            "results",
        )

    if "summary" in payload:
        summary = payload["summary"]
        if not isinstance(summary, Mapping):
            raise ValueError("summary must be an object")

        config = _normalize_config(payload.get("config"), "summary")
        config.update(_normalize_config(summary.get("config"), "summary"))
        finals = summary.get("final", {})
        if not isinstance(finals, Mapping):
            raise ValueError("summary final results must be an object")

        records = []
        for method, metrics in finals.items():
            if not isinstance(metrics, Mapping):
                raise ValueError(f"summary metrics for {method!r} must be an object")
            record = dict(metrics)
            record["method"] = method
            records.append(record)
        return ResultBundle(records, config, "summary")

    raise ValueError("unrecognized result schema")


def _load_companion_config(result_path: Path) -> Dict[str, Any]:
    candidates = (
        result_path.with_name(f"{result_path.stem}_config.json"),
        result_path.with_name("config.yaml"),
        result_path.with_name("config.yml"),
    )
    merged: Dict[str, Any] = {}
    for config_path in candidates:
        if not config_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            if config_path.suffix == ".json":
                config = json.load(handle)
            else:
                config = yaml.safe_load(handle)
        # Keep the historical priority of ``*_config.json`` over YAML while
        # allowing later files to supplement fields that are absent from it.
        for key, value in _normalize_config(config, "companion").items():
            merged.setdefault(key, value)
    return merged


def load_result_bundle(path: Any) -> ResultBundle:
    """Load a result JSON file and supplement missing config from companion files."""
    result_path = Path(path)
    with result_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    bundle = normalize_result_payload(payload)
    companion_config = _load_companion_config(result_path)
    config = {**companion_config, **bundle.config}
    return ResultBundle(bundle.records, config, bundle.schema, result_path)


def primary_metric(record: Mapping[str, Any]) -> tuple[Optional[str], Any]:
    """Return the primary metric key and value from one normalized record."""
    candidates = dict(record)
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping):
        # Records from different runners use both layouts.  Treat top-level
        # values as defaults and let the normalized nested block override them.
        candidates.update(metrics)

    accuracy = candidates.get("accuracy")
    if "perplexity" in candidates and accuracy in (None, 0):
        return "perplexity", candidates["perplexity"]

    for key in (
        "accuracy",
        "top1_accuracy",
        "perplexity",
        "pearson",
        "final_loss",
        "val_loss",
        "loss",
        "metric_value",
    ):
        if key in candidates:
            return key, candidates[key]
    return None, None
