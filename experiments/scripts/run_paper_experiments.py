"""Run declaration-driven, paired residual-recovery experiments for the paper."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from typing import Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.data import cache_batches, get_data_loaders  # noqa: E402
from experiments.lib.models import load_model  # noqa: E402
from experiments.lib.paper_baselines import validate_claim_gate  # noqa: E402
from experiments.lib.paper_manifest import load_paper_manifest  # noqa: E402
from experiments.lib.paper_results import (  # noqa: E402
    SCHEMA_VERSION,
    JobResultStore,
    SuiteResultStore,
    records_from,
)
from experiments.lib.paper_runner import (  # noqa: E402
    aggregate_job_results,
    plan_jobs,
    run_method_trajectory,
)
from experiments.lib.residual_runtime import (  # noqa: E402
    LoadedTrainingCheckpoint,
    batch_hash,
    configure_hf_offline,
    empty_device_cache,
    load_training_checkpoint,
    set_seed,
    sha256_file,
    write_json,
)
from experiments.lib.residual_training import partition_repeated_seed_batches  # noqa: E402


def _csv(values: str | None, cast) -> tuple:
    if values is None:
        return ()
    raw = [value.strip() for value in values.split(",")]
    if any(not value for value in raw):
        raise ValueError("Comma-separated selections must not contain empty values")
    converted = tuple(cast(value) for value in raw)
    if len(set(converted)) != len(converted):
        raise ValueError(f"Comma-separated selections must not contain duplicates: {values}")
    return converted


def _run_git_bytes(arguments: Sequence[str]) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def _source_provenance(manifest_path: Path) -> dict[str, object]:
    pathspecs = ["*.py", "*.yaml", "*.yml", "*.toml", "*.sh"]
    listed = _run_git_bytes(
        [
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *pathspecs,
        ]
    )
    source_paths = []
    excluded_roots = {"results", "data", "checkpoints", "archive"}
    source_suffixes = {".py", ".yaml", ".yml", ".toml", ".sh"}
    for raw_path in (path for path in listed.split(b"\0") if path):
        relative_text = raw_path.decode("utf-8", errors="surrogateescape")
        parsed = Path(relative_text)
        if parsed.parts and parsed.parts[0] in excluded_roots:
            continue
        if parsed.suffix.lower() in source_suffixes and (ROOT / parsed).is_file():
            source_paths.append((raw_path, parsed))
    state_digest = hashlib.sha256()
    changed = _run_git_bytes(["diff", "--name-only", "-z", "HEAD", "--", *pathspecs])
    untracked = _run_git_bytes(
        ["ls-files", "--others", "--exclude-standard", "-z", "--", *pathspecs]
    )
    dirty_paths = []
    for raw_path in changed.split(b"\0") + untracked.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="replace").replace("\\", "/")
        parsed = Path(relative)
        if parsed.parts and parsed.parts[0] in excluded_roots:
            continue
        if parsed.suffix.lower() in source_suffixes:
            dirty_paths.append(relative)
    dirty_paths = sorted(set(dirty_paths))
    for raw_path, relative_path in sorted(source_paths):
        source_path = ROOT / relative_path
        state_digest.update(len(raw_path).to_bytes(8, "big"))
        state_digest.update(raw_path)
        state_digest.update(bytes.fromhex(sha256_file(source_path)))
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git_commit": _run_git_bytes(["rev-parse", "HEAD"]).decode("ascii").strip(),
        "git_dirty": bool(dirty_paths),
        "dirty_source_paths": dirty_paths,
        "source_file_count": len(source_paths),
        "source_state_sha256": state_digest.hexdigest(),
    }


def _remap(path: Path, declared_root: Path, override: Path | None) -> Path:
    if override is None:
        return path
    try:
        relative = path.relative_to(declared_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Cannot remap {path}; it is outside {declared_root}") from exc
    return (override.resolve() / relative).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments/configs/paper_experiments.yaml",
    )
    parser.add_argument("--workload", help="Comma-separated workload names")
    parser.add_argument("--prune-ratio", help="Comma-separated declared ratios")
    parser.add_argument("--recoveries", help="Comma-separated declared K values")
    parser.add_argument("--seeds", help="Comma-separated declared seeds")
    parser.add_argument("--methods", help="Comma-separated manifest method names")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Remap declared trusted local checkpoints beneath this root.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Remap declared datasets beneath this root.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_runs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a strictly compatible suite and skip verified completed records.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_device(device: str) -> str:
    try:
        parsed = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device: {device}") from exc
    if parsed.type not in {"cpu", "cuda"}:
        raise ValueError(f"Paper experiments support only CPU or CUDA, got {device}")
    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if parsed.index is not None and parsed.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {device}")
    return str(parsed)


@dataclass(frozen=True)
class WorkloadContext:
    """Read-only checkpoint and batch inputs shared by one workload's jobs."""

    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint: LoadedTrainingCheckpoint
    training_pool: Sequence[Mapping[str, torch.Tensor]]
    eval_batches: Sequence[Mapping[str, torch.Tensor]]
    data_identity: Mapping[str, object]


def _job_template(
    job,
    contracts,
    provenance: dict[str, object],
    context: WorkloadContext,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "started_at": _utc_now(),
        "provenance": {
            **provenance,
            "checkpoint": str(context.checkpoint_path),
            "checkpoint_sha256": context.checkpoint_sha256,
            "checkpoint_step": context.checkpoint.step,
        },
        "config": job.to_result_dict(),
        "method_contracts": [contract.to_result_dict() for contract in contracts],
        "training_pool": {
            "count": len(context.training_pool),
            "sha256": batch_hash(context.training_pool),
        },
        "evaluation_batches": {
            "count": len(context.eval_batches),
            "sha256": batch_hash(context.eval_batches),
        },
        "batch_plans": {},
        "results": [],
    }


def _load_workload_context(
    workload,
    *,
    checkpoint_root: Path | None,
    data_root: Path | None,
    declared_checkpoint_root: Path,
    declared_data_root: Path,
    expected_data_identity: Mapping[str, object] | None,
) -> WorkloadContext:
    """Load and verify immutable inputs once for all jobs of one workload."""
    checkpoint_path = _remap(workload.checkpoint, declared_checkpoint_root, checkpoint_root)
    data_dir = _remap(workload.data_dir, declared_data_root, data_root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint_stat = checkpoint_path.stat()
    checkpoint_identity = (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = load_training_checkpoint(checkpoint_path)
    loaded_stat = checkpoint_path.stat()
    if (loaded_stat.st_size, loaded_stat.st_mtime_ns) != checkpoint_identity:
        raise RuntimeError(f"Checkpoint changed while it was being loaded: {checkpoint_path}")
    if checkpoint.step != workload.checkpoint_step:
        raise ValueError(
            f"{workload.name}: checkpoint reports step {checkpoint.step}; "
            f"manifest requires {workload.checkpoint_step}"
        )
    set_seed(0)
    train_loader, val_loader, task_type = get_data_loaders(
        workload.model,
        workload.dataset,
        workload.batch_size,
        workload.seq_length,
        data_dir=str(data_dir),
    )
    if task_type != workload.task_type:
        raise RuntimeError("Data loader task type does not match the manifest")
    training_pool = cache_batches(train_loader, workload.train_pool_batches, workload.task_type)
    eval_batches = cache_batches(val_loader, workload.eval_batches, workload.task_type)
    if len(training_pool) != workload.train_pool_batches:
        raise RuntimeError(
            f"{workload.name}: training loader provided {len(training_pool)} cached batches; "
            f"expected {workload.train_pool_batches}"
        )
    if len(eval_batches) != workload.eval_batches:
        raise RuntimeError(
            f"{workload.name}: evaluation loader provided {len(eval_batches)} cached batches; "
            f"expected {workload.eval_batches}"
        )
    data_identity = {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint.step,
        "training_pool": {
            "count": len(training_pool),
            "sha256": batch_hash(training_pool),
        },
        "evaluation_batches": {
            "count": len(eval_batches),
            "sha256": batch_hash(eval_batches),
        },
    }
    if expected_data_identity is not None and expected_data_identity != data_identity:
        raise ValueError(f"{workload.name}: cached data identity changed across suite jobs")
    del train_loader, val_loader
    return WorkloadContext(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint=checkpoint,
        training_pool=training_pool,
        eval_batches=eval_batches,
        data_identity=data_identity,
    )


def _run_job(
    job,
    contracts,
    *,
    args: argparse.Namespace,
    provenance: dict[str, object],
    context: WorkloadContext,
) -> Path:
    workload = job.workload

    model_factory = lambda workload=workload: load_model(
        workload.model,
        pretrained=False,
        checkpoint_path=None,
        device="cpu",
        dataset_name=workload.dataset,
    )[0]
    job_path = Path(args.output_dir) / f"{job.job_id}.json"
    store = JobResultStore.open(
        job_path,
        _job_template(job, contracts, provenance, context),
        metric=workload.metric,
        resume=args.resume,
    )
    try:
        for seed in job.seeds:
            batch_plan = partition_repeated_seed_batches(
                context.training_pool,
                workload.total_steps,
                job.recovery_count,
                workload.hvp_batches,
                seed,
            )
            store.ensure_batch_plan(
                seed,
                {
                    "selected_pool_indices": batch_plan.selected_pool_indices,
                    **batch_plan.data_hashes(),
                },
            )
            for contract in contracts:
                if store.has_result(seed, contract.name):
                    continue
                record = run_method_trajectory(
                    job,
                    contract,
                    seed,
                    batch_plan,
                    model_factory,
                    context.checkpoint.model_state,
                    context.checkpoint.optimizer_state,
                    context.eval_batches,
                    args.device,
                )
                store.append_result(record)
        aggregate = aggregate_job_results(records_from(store), workload.metric)
        store.finalize(aggregate, finished_at=_utc_now())
        return job_path
    finally:
        empty_device_cache(args.device)


def _validate_completed_checkpoint_files(
    jobs,
    suite: SuiteResultStore,
    *,
    checkpoint_root: Path | None,
    declared_checkpoint_root: Path,
) -> None:
    """Verify checkpoints only for workloads that bypass context loading entirely."""
    jobs_by_workload: dict[str, list] = {}
    for job in jobs:
        jobs_by_workload.setdefault(job.workload.name, []).append(job)

    for workload_jobs in jobs_by_workload.values():
        if not all(suite.has_completed_job(job.job_id) for job in workload_jobs):
            continue
        workload = workload_jobs[0].workload
        identity = suite.expected_data_identity(workload.name)
        if not isinstance(identity, Mapping):
            raise ValueError(f"{workload.name}: completed job has no data identity")
        expected_digest = identity.get("checkpoint_sha256")
        expected_step = identity.get("checkpoint_step")
        if not isinstance(expected_digest, str) or not expected_digest:
            raise ValueError(f"{workload.name}: completed job has invalid checkpoint digest")
        if (
            isinstance(expected_step, bool)
            or not isinstance(expected_step, int)
            or expected_step < 0
        ):
            raise ValueError(f"{workload.name}: completed job has invalid checkpoint step")
        if expected_step != workload.checkpoint_step:
            raise ValueError(
                f"{workload.name}: completed job checkpoint step {expected_step} "
                f"does not match manifest step {workload.checkpoint_step}"
            )

        checkpoint_path = _remap(
            workload.checkpoint,
            declared_checkpoint_root,
            checkpoint_root,
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        stat_before = checkpoint_path.stat()
        actual_digest = sha256_file(checkpoint_path)
        stat_after = checkpoint_path.stat()
        if (stat_before.st_size, stat_before.st_mtime_ns) != (
            stat_after.st_size,
            stat_after.st_mtime_ns,
        ):
            raise RuntimeError(f"Checkpoint changed while it was being hashed: {checkpoint_path}")
        if actual_digest != expected_digest:
            raise ValueError(f"{workload.name}: checkpoint digest changed since the completed job")


def _preflight_outputs(output_dir: Path, jobs, *, resume: bool) -> None:
    suite_path = output_dir / "suite_manifest.json"
    job_paths = [output_dir / f"{job.job_id}.json" for job in jobs]
    if resume:
        if not suite_path.is_file():
            raise FileNotFoundError(f"Cannot resume missing suite state: {suite_path}")
        return
    existing = [path for path in [suite_path, *job_paths] if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Experiment outputs already exist: {formatted}. "
            "Use --resume or a new output directory."
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = load_paper_manifest(args.manifest)
    jobs = plan_jobs(
        manifest.workloads,
        workload_names=_csv(args.workload, str),
        prune_ratios=_csv(args.prune_ratio, float),
        recovery_counts=_csv(args.recoveries, int),
        seeds=_csv(args.seeds, int),
    )
    contracts = list(manifest.methods)
    if args.methods:
        requested = _csv(args.methods, str)
        by_name = {contract.name: contract for contract in contracts}
        unknown = set(requested) - set(by_name)
        if unknown:
            raise ValueError(f"Methods are not declared in the manifest: {sorted(unknown)}")
        contracts = [by_name[name] for name in requested]

    for gate_name, gate in manifest.claim_gates.items():
        if gate.get("enabled", True):
            try:
                validate_claim_gate(contracts, gate)
            except ValueError as exc:
                raise ValueError(
                    f"Selected methods do not satisfy enabled claim gate {gate_name!r}: {exc}"
                ) from exc

    plan = {
        "jobs": [job.to_result_dict() for job in jobs],
        "methods": [contract.to_result_dict() for contract in contracts],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        return
    args.device = _validate_device(args.device)

    configure_hf_offline()
    provenance = _source_provenance(manifest.path)
    suite_path = args.output_dir / "suite_manifest.json"
    _preflight_outputs(args.output_dir, jobs, resume=args.resume)
    suite_template = {
        "schema_version": SCHEMA_VERSION,
        "status": "started",
        "started_at": _utc_now(),
        "provenance": provenance,
        "plan": plan,
        "planned_jobs": [job.job_id for job in jobs],
        "completed_jobs": {},
        "data_identities": {},
    }
    suite = SuiteResultStore.open(suite_path, suite_template, resume=args.resume)
    declared_checkpoint_root = ROOT / "checkpoints"
    declared_data_root = ROOT / "data"
    if args.resume:
        _validate_completed_checkpoint_files(
            jobs,
            suite,
            checkpoint_root=args.checkpoint_root,
            declared_checkpoint_root=declared_checkpoint_root,
        )
    if suite.complete:
        return

    for workload_name, workload_jobs_iter in groupby(
        jobs,
        key=lambda job: job.workload.name,
    ):
        workload_jobs = list(workload_jobs_iter)
        pending_jobs = [job for job in workload_jobs if not suite.has_completed_job(job.job_id)]
        if not pending_jobs:
            continue
        workload = pending_jobs[0].workload
        context = _load_workload_context(
            workload,
            checkpoint_root=args.checkpoint_root,
            data_root=args.data_root,
            declared_checkpoint_root=declared_checkpoint_root,
            declared_data_root=declared_data_root,
            expected_data_identity=suite.expected_data_identity(workload_name),
        )
        try:
            for job in pending_jobs:
                job_path = _run_job(
                    job,
                    contracts,
                    args=args,
                    provenance=provenance,
                    context=context,
                )
                suite.record_completed_job(job.job_id, job_path)
        finally:
            del context
            gc.collect()
            empty_device_cache(args.device)

    suite.finalize(finished_at=_utc_now())


if __name__ == "__main__":
    main()
