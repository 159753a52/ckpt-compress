"""Run the audited full-weight pruning-only comparison."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.data import cache_batches, get_data_loaders, get_task_type
from experiments.lib.evaluation import compute_quality_drop
from experiments.lib.fullweight_comparison import (
    FULLWEIGHT_METHOD_CONTRACTS,
    FULLWEIGHT_METHOD_IDS,
    apply_zero_masks_to_state,
    assemble_fullweight_comparison,
    validate_search_result_provenance,
)
from experiments.lib.models import get_model_type, load_model
from experiments.lib.residual_runtime import (
    batch_hash,
    configure_hf_offline,
    empty_device_cache,
    evaluate_task,
    load_training_checkpoint,
    peak_memory_bytes,
    reset_peak_memory,
    set_seed,
    sha256_file,
    synchronize_device,
    write_json,
)
from experiments.lib.residual_scoring import (
    compute_block_first_order_scores,
    compute_block_taylor_scores,
    eligible_layers,
    transformer_layers,
)
from experiments.scripts.run_paper_experiments import _source_provenance

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FullWeightWorkload:
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

    def to_result_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "checkpoint": str(self.checkpoint),
            "data_dir": str(self.data_dir),
            "seeds": list(self.seeds),
        }


@dataclass(frozen=True)
class FullWeightManifest:
    path: Path
    experiment_scope: str
    direct_weight_prune_ratios: tuple[float, ...]
    hvp_batches: int
    eval_batches: int
    batch_size: int
    seq_length: int
    protect_fraction: float
    inshrinkerator_search_json: Path
    workloads: tuple[FullWeightWorkload, ...]


def _resolve_path(value: object, manifest_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest path must be a non-empty string, got {value!r}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent.parent.parent / path
    return path.resolve()


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_float(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or minimum is not None and normalized < minimum:
        raise ValueError(f"{label} must be a finite number")
    return normalized


def _float_tuple(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    values = tuple(_finite_float(item, label) for item in value)
    if any(not 0.0 < item < 1.0 for item in values):
        raise ValueError(f"{label} must contain values in (0, 1)")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")
    return values


def load_fullweight_manifest(path: Path) -> FullWeightManifest:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if (
        not isinstance(payload, Mapping)
        or isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 1
    ):
        raise ValueError("Full-weight manifest schema_version must be 1")
    if payload.get("experiment_scope") != "full_weight_pruning_only":
        raise ValueError(
            "Full-weight manifest must declare experiment_scope=full_weight_pruning_only"
        )
    raw_workloads = payload.get("workloads")
    if not isinstance(raw_workloads, list) or len(raw_workloads) != 1:
        raise ValueError("Full-weight manifest must declare exactly one workload")
    raw = raw_workloads[0]
    if not isinstance(raw, Mapping):
        raise ValueError("Full-weight workload must be an object")
    name = raw.get("name")
    model = raw.get("model")
    dataset = raw.get("dataset")
    if not all(isinstance(value, str) and value for value in (name, model, dataset)):
        raise ValueError("Full-weight workload name, model, and dataset are required")
    family = get_model_type(model)
    task_type = get_task_type(dataset)
    metric = raw.get("metric")
    metrics_by_task = {
        "lm": {"loss", "perplexity"},
        "cls": {"accuracy"},
        "reg": {"loss", "pearson"},
    }
    if not isinstance(metric, str) or metric not in metrics_by_task.get(task_type, set()):
        raise ValueError(f"{name}: metric {metric!r} is incompatible with {task_type}")
    raw_seeds = raw.get("seeds")
    if (
        not isinstance(raw_seeds, list)
        or not raw_seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in raw_seeds
        )
        or len(set(raw_seeds)) != len(raw_seeds)
    ):
        raise ValueError(f"{name}: seeds must be unique non-negative integers")
    workload = FullWeightWorkload(
        name=name,
        model=model,
        model_family=family,
        dataset=dataset,
        task_type=task_type,
        checkpoint=_resolve_path(raw.get("checkpoint"), path),
        checkpoint_step=_positive_int(raw.get("checkpoint_step"), f"{name}.checkpoint_step"),
        data_dir=_resolve_path(raw.get("data_dir"), path),
        metric=metric,
        seeds=tuple(raw_seeds),
    )
    protect_fraction = _finite_float(payload.get("protect_fraction"), "protect_fraction")
    if not 0.0 <= protect_fraction < 1.0:
        raise ValueError("protect_fraction must be in [0, 1)")
    search_json = _resolve_path(payload.get("inshrinkerator_search_json"), path)
    return FullWeightManifest(
        path=path,
        experiment_scope=payload["experiment_scope"],
        direct_weight_prune_ratios=_float_tuple(
            payload.get("direct_weight_prune_ratios"),
            "direct_weight_prune_ratios",
        ),
        hvp_batches=_positive_int(payload.get("hvp_batches"), "hvp_batches"),
        eval_batches=_positive_int(payload.get("eval_batches"), "eval_batches"),
        batch_size=_positive_int(payload.get("batch_size"), "batch_size"),
        seq_length=_positive_int(payload.get("seq_length"), "seq_length"),
        protect_fraction=protect_fraction,
        inshrinkerator_search_json=search_json,
        workloads=(workload,),
    )


def _csv(value: str | None, cast) -> tuple:
    if value is None:
        return ()
    raw_values = [item.strip() for item in value.split(",")]
    if not raw_values or any(not item for item in raw_values):
        raise ValueError("Comma-separated selections must not contain empty values")
    converted = tuple(cast(item) for item in raw_values)
    if len(set(converted)) != len(converted):
        raise ValueError("Comma-separated selections must not contain duplicates")
    return converted


def _select_manifest(
    manifest: FullWeightManifest,
    workload_name: str | None,
    seed: int,
    requested_ratios: Sequence[float],
) -> tuple[FullWeightWorkload, tuple[float, ...]]:
    workload = manifest.workloads[0]
    if workload_name is not None and workload_name != workload.name:
        raise ValueError(f"Unknown workload: {workload_name}")
    if seed not in workload.seeds:
        raise ValueError(f"Seed {seed} is not declared for workload {workload.name}")
    ratios = tuple(requested_ratios) or manifest.direct_weight_prune_ratios
    if not set(ratios).issubset(manifest.direct_weight_prune_ratios):
        raise ValueError("Requested prune ratio is not declared in the manifest")
    return workload, ratios


def _validate_device(device: str) -> str:
    try:
        parsed = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device: {device}") from exc
    if parsed.type not in {"cpu", "cuda"}:
        raise ValueError("Full-weight comparison supports only CPU or CUDA")
    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if parsed.index is not None and parsed.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {device}")
    return str(parsed)


def _load_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value is not allowed: {value}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ratio_key(ratio: float) -> str:
    return float(ratio).hex().replace("0x", "").replace(".", "p").replace("+", "")


def _ratio_path(output_dir: Path, workload: FullWeightWorkload, ratio: float) -> Path:
    return output_dir / f"{workload.name}_p{_ratio_key(ratio)}.json"


def _canonical(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


class FullWeightSuiteStore:
    """Minimal digest-bound suite state for independent full-weight runs."""

    def __init__(self, path: Path, payload: dict[str, object]):
        self.path = path
        self.payload = payload

    @classmethod
    def open(
        cls,
        path: Path,
        template: Mapping[str, object],
        *,
        resume: bool,
    ) -> "FullWeightSuiteStore":
        path = Path(path)
        if path.exists():
            if not resume:
                raise FileExistsError(f"Suite output already exists: {path}; use --resume")
            payload = _load_json(path)
            for key in (
                "schema_version",
                "experiment_scope",
                "plan",
                "provenance",
                "planned_ratios",
            ):
                if payload.get(key) != _canonical(template.get(key)):
                    raise ValueError(f"Suite state is incompatible: {key}")
            if not isinstance(payload.get("completed_ratios"), dict):
                raise ValueError("Suite state has invalid completed_ratios")
            return cls(path, payload)
        if resume:
            raise FileNotFoundError(f"Cannot resume missing suite state: {path}")
        payload = dict(_canonical(template))
        payload["started_at"] = _utc_now()
        payload["status"] = "started"
        write_json(path, payload)
        return cls(path, payload)

    def completed_path(self, ratio_key: str, expected_payload: Mapping[str, object]) -> Path | None:
        completed = self.payload["completed_ratios"]
        if not isinstance(completed, dict):
            raise ValueError("Suite state has invalid completed_ratios")
        raw_entry = completed.get(ratio_key)
        if raw_entry is None:
            return None
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Completed ratio {ratio_key} has malformed digest entry")
        relative = raw_entry.get("path")
        expected_digest = raw_entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_digest, str):
            raise ValueError(f"Completed ratio {ratio_key} has malformed digest entry")
        result_path = (self.path.parent / relative).resolve()
        try:
            result_path.relative_to(self.path.parent.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Completed ratio {ratio_key} points outside the suite directory"
            ) from exc
        if not result_path.is_file() or sha256_file(result_path) != expected_digest:
            raise ValueError(f"Completed ratio {ratio_key} digest does not match its result file")
        payload = _load_json(result_path)
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("experiment_scope") != expected_payload.get("experiment_scope")
            or payload.get("status") != "complete"
            or payload.get("config") != expected_payload.get("config")
            or not isinstance(payload.get("results"), list)
            or len(payload["results"]) != len(FULLWEIGHT_METHOD_IDS)
        ):
            raise ValueError(f"Completed ratio {ratio_key} is not a compatible complete record")
        if payload.get("provenance") != expected_payload.get("provenance"):
            raise ValueError(f"Completed ratio {ratio_key} provenance does not match the suite")
        return result_path

    def record_completed(self, ratio_key: str, result_path: Path) -> None:
        completed = self.payload.get("completed_ratios")
        planned = self.payload.get("planned_ratios")
        if not isinstance(completed, dict) or not isinstance(planned, list):
            raise ValueError("Suite state has invalid completion metadata")
        if ratio_key in completed:
            raise ValueError(f"Ratio {ratio_key} is already recorded")
        relative = result_path.resolve().relative_to(self.path.parent.resolve())
        candidate = copy.deepcopy(self.payload)
        candidate["completed_ratios"][ratio_key] = {
            "path": str(relative).replace("\\", "/"),
            "sha256": sha256_file(result_path),
        }
        write_json(self.path, candidate)
        self.payload = candidate

    def finalize(self) -> None:
        completed = self.payload.get("completed_ratios")
        planned = self.payload.get("planned_ratios")
        if not isinstance(completed, dict) or not isinstance(planned, list):
            raise ValueError("Suite state has invalid completion metadata")
        if set(completed) != set(planned):
            raise ValueError("Cannot finalize suite with incomplete ratios")
        candidate = copy.deepcopy(self.payload)
        candidate["status"] = "complete"
        candidate["finished_at"] = _utc_now()
        write_json(self.path, candidate)
        self.payload = candidate


def _result_template(
    *,
    workload: FullWeightWorkload,
    manifest: FullWeightManifest,
    ratio: float,
    seed: int,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "experiment_scope": manifest.experiment_scope,
        "config": {
            "workload": workload.to_result_dict(),
            "prune_ratio": ratio,
            "seed": seed,
            "protect_fraction": manifest.protect_fraction,
        },
        "provenance": dict(provenance),
    }


def _evaluate_state(
    base_model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    eval_batches,
    task_type: str,
    device: str,
) -> tuple[dict[str, float | int], float, int]:
    reset_peak_memory(device)
    started = time.perf_counter()
    model = copy.deepcopy(base_model)
    model.load_state_dict(state, strict=True)
    model.to(device)
    metrics = evaluate_task(model, eval_batches, task_type, device)
    synchronize_device(device)
    wall_seconds = time.perf_counter() - started
    peak_bytes = peak_memory_bytes(device)
    del model
    empty_device_cache(device)
    return metrics, wall_seconds, peak_bytes


def _compute_fullweight_score_families(
    model: torch.nn.Module,
    scoring_batches,
    layers,
    blocks,
    parameters: Mapping[str, torch.Tensor],
    device: str,
    task_type: str,
    model_family: str,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, object]]:
    """Compute all scores against the same current full-weight theta probe."""
    first_order_scores, first_order_scoring = compute_block_first_order_scores(
        model,
        scoring_batches,
        layers,
        parameters,
        device,
        task_type=task_type,
        model_family=model_family,
        block_parameter_names=blocks,
    )
    taylor_scores, taylor_scoring = compute_block_taylor_scores(
        model,
        scoring_batches,
        layers,
        parameters,
        device,
        return_components=False,
        task_type=task_type,
        model_family=model_family,
        block_parameter_names=blocks,
    )
    magnitude_scores = {name: value.abs() for name, value in parameters.items()}
    return (
        {
            "magnitude": magnitude_scores,
            "first_order": first_order_scores,
            "taylor_hvp": taylor_scores,
        },
        {
            "first_order": first_order_scoring,
            "taylor_hvp": taylor_scoring,
        },
    )


def _method_contracts() -> list[dict[str, str]]:
    return [
        FULLWEIGHT_METHOD_CONTRACTS[method_id].to_result_dict()
        for method_id in FULLWEIGHT_METHOD_IDS
    ]


def _run_comparison(
    *,
    manifest: FullWeightManifest,
    workload: FullWeightWorkload,
    seed: int,
    ratios: Sequence[float],
    search_path: Path,
    device: str,
    output_dir: Path,
    resume: bool,
) -> None:
    configure_hf_offline()
    set_seed(seed)
    checkpoint_sha256 = sha256_file(workload.checkpoint)
    checkpoint = load_training_checkpoint(workload.checkpoint)
    if checkpoint.step != workload.checkpoint_step:
        raise ValueError(
            f"{workload.name}: checkpoint reports step {checkpoint.step}; "
            f"manifest requires {workload.checkpoint_step}"
        )
    train_loader, eval_loader, task_type = get_data_loaders(
        workload.model,
        workload.dataset,
        manifest.batch_size,
        manifest.seq_length,
        data_dir=str(workload.data_dir),
    )
    if task_type != workload.task_type:
        raise ValueError("Data loader task type does not match the full-weight manifest")
    scoring_batches = cache_batches(train_loader, manifest.hvp_batches, task_type)
    evaluation_batches = cache_batches(eval_loader, manifest.eval_batches, task_type)
    if len(scoring_batches) != manifest.hvp_batches:
        raise RuntimeError("Training loader provided fewer scoring batches than declared")
    if len(evaluation_batches) != manifest.eval_batches:
        raise RuntimeError("Evaluation loader provided fewer batches than declared")

    source_provenance = _source_provenance(manifest.path)
    source_digest = source_provenance.get("source_state_sha256")
    if not isinstance(source_digest, str):
        raise RuntimeError("Source provenance did not return a digest")
    search_payload = _load_json(search_path)
    validate_search_result_provenance(
        search_payload,
        model=workload.model,
        dataset=workload.dataset,
        seed=seed,
        checkpoint=workload.checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        source_digest=source_digest,
        scoring_batch_sha256=batch_hash(scoring_batches),
        evaluation_batch_sha256=batch_hash(evaluation_batches),
        scoring_batch_count=len(scoring_batches),
        evaluation_batch_count=len(evaluation_batches),
        protect_fraction=manifest.protect_fraction,
    )

    base_model, model_family = load_model(
        workload.model,
        pretrained=False,
        checkpoint_path=None,
        device="cpu",
        dataset_name=workload.dataset,
    )
    base_model.load_state_dict(checkpoint.model_state, strict=True)
    base_model.eval()
    base_state = {
        name: value.detach().cpu().clone() for name, value in base_model.state_dict().items()
    }
    layers = eligible_layers(base_model, model_family)
    blocks = transformer_layers(base_model, model_family)
    eligible_names = [name for layer in layers for name in layer]
    named_parameters = dict(base_model.named_parameters())
    parameters = {
        name: named_parameters[name].detach().float().cpu().clone() for name in eligible_names
    }

    scoring_model = copy.deepcopy(base_model).to(device)
    score_families, score_metadata = _compute_fullweight_score_families(
        scoring_model,
        scoring_batches,
        layers,
        blocks,
        parameters,
        device,
        task_type=task_type,
        model_family=model_family,
    )
    del scoring_model
    empty_device_cache(device)
    first_order_scoring = score_metadata["first_order"]
    taylor_scoring = score_metadata["taylor_hvp"]
    baseline_metrics, baseline_wall, baseline_peak = _evaluate_state(
        base_model,
        base_state,
        evaluation_batches,
        task_type,
        device,
    )
    provenance = {
        "manifest": str(manifest.path),
        "manifest_sha256": sha256_file(manifest.path),
        "checkpoint": str(workload.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint.step,
        "search_json": str(search_path),
        "search_json_sha256": sha256_file(search_path),
        "source_digest": source_digest,
        "scoring_batch_sha256": batch_hash(scoring_batches),
        "evaluation_batch_sha256": batch_hash(evaluation_batches),
        "scoring_batch_count": len(scoring_batches),
        "evaluation_batch_count": len(evaluation_batches),
    }
    plan = {
        "workload": workload.to_result_dict(),
        "seed": seed,
        "ratios": list(ratios),
        "methods": _method_contracts(),
        "experiment_scope": manifest.experiment_scope,
    }
    suite_template = {
        "schema_version": SCHEMA_VERSION,
        "experiment_scope": manifest.experiment_scope,
        "plan": plan,
        "provenance": provenance,
        "planned_ratios": [_ratio_key(ratio) for ratio in ratios],
        "completed_ratios": {},
    }
    output_dir = Path(output_dir).resolve()
    result_paths = {_ratio_key(ratio): _ratio_path(output_dir, workload, ratio) for ratio in ratios}
    if not resume:
        existing = [
            path
            for path in [output_dir / "suite_manifest.json", *result_paths.values()]
            if path.exists()
        ]
        if existing:
            raise FileExistsError(f"Experiment outputs already exist: {existing}")
    suite = FullWeightSuiteStore.open(
        output_dir / "suite_manifest.json",
        suite_template,
        resume=resume,
    )

    for ratio in ratios:
        key = _ratio_key(ratio)
        result_template = _result_template(
            workload=workload,
            manifest=manifest,
            ratio=ratio,
            seed=seed,
            provenance=provenance,
        )
        completed = suite.completed_path(key, result_template) if resume else None
        if completed is not None:
            continue
        result_path = result_paths[key]
        if result_path.exists():
            raise FileExistsError(f"Refusing to overwrite incomplete ratio result: {result_path}")
        ratio_started = time.perf_counter()
        methods = []
        selected_metric = search_payload["best_metric"]
        assemblies = assemble_fullweight_comparison(
            parameters,
            layers,
            score_families,
            search_payload,
            ratio,
            model_family=model_family,
        )
        for assembly in assemblies:
            zeroed_state = apply_zero_masks_to_state(base_state, assembly.masks)
            metrics, wall_seconds, peak_bytes = _evaluate_state(
                base_model,
                zeroed_state,
                evaluation_batches,
                task_type,
                device,
            )
            result = assembly.to_result_dict()
            result.update(
                {
                    "metrics": metrics,
                    "relative_degradation_pct": compute_quality_drop(
                        baseline_metrics,
                        metrics,
                        task_type,
                    ),
                    "wall_seconds": wall_seconds,
                    "peak_gpu_memory_bytes": peak_bytes,
                    "direct_zeroing": True,
                    "protect_fraction": manifest.protect_fraction,
                    "protection_applied": False,
                    "scoring": (
                        taylor_scoring
                        if assembly.method_id == "dacp_fullweight"
                        else first_order_scoring
                    ),
                }
            )
            methods.append(result)
        ratio_payload = {
            **result_template,
            "started_at": _utc_now(),
            "finished_at": _utc_now(),
            "baseline": {
                "metrics": baseline_metrics,
                "wall_seconds": baseline_wall,
                "peak_gpu_memory_bytes": baseline_peak,
            },
            "score_provenance": {
                "magnitude": {
                    "score_kind": "residual_magnitude",
                    "source": "current_full_weight_theta",
                },
                "first_order": first_order_scoring,
                "taylor_hvp": taylor_scoring,
                "selected_search_metric": selected_metric,
            },
            "results": methods,
            "wall_seconds": time.perf_counter() - ratio_started,
        }
        write_json(result_path, ratio_payload)
        suite.record_completed(key, result_path)
    suite.finalize()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments/configs/paper_fullweight_inshrinkerator.yaml",
    )
    parser.add_argument("--workload")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prune-ratios", "--prune_ratios", dest="prune_ratios")
    parser.add_argument("--search-json", "--search_json", dest="search_json", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=ROOT / "results/paper_runs/fullweight_inshrinkerator",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    manifest = load_fullweight_manifest(args.manifest)
    requested_ratios = _csv(args.prune_ratios, float)
    workload, ratios = _select_manifest(manifest, args.workload, args.seed, requested_ratios)
    search_path = (
        args.search_json.resolve()
        if args.search_json is not None
        else manifest.inshrinkerator_search_json
    )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "experiment_scope": manifest.experiment_scope,
        "workload": workload.to_result_dict(),
        "seed": args.seed,
        "direct_weight_prune_ratios": list(ratios),
        "methods": _method_contracts(),
        "protect_fraction": manifest.protect_fraction,
        "inshrinkerator_search_json": str(search_path),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        return
    if not args.resume:
        output_dir = args.output_dir.resolve()
        candidate_paths = [
            output_dir / "suite_manifest.json",
            *(_ratio_path(output_dir, workload, ratio) for ratio in ratios),
        ]
        existing = [path for path in candidate_paths if path.exists()]
        if existing:
            raise FileExistsError(f"Experiment outputs already exist: {existing}")
    device = _validate_device(args.device)
    _run_comparison(
        manifest=manifest,
        workload=workload,
        seed=args.seed,
        ratios=ratios,
        search_path=search_path,
        device=device,
        output_dir=args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
