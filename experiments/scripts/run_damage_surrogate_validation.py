"""Run the mask-level damage-surrogate validation at one recovery point."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.damage_surrogate_validation import (  # noqa: E402
    MaskCandidate,
    validate_damage_surrogate,
)
from experiments.lib.data import cache_batches, get_data_loaders  # noqa: E402
from experiments.lib.models import load_model  # noqa: E402
from experiments.lib.paper_manifest import load_paper_manifest  # noqa: E402
from experiments.lib.paper_runner import plan_jobs, recovery_protocol  # noqa: E402
from experiments.lib.residual_masks import global_mask, layer_masks  # noqa: E402
from experiments.lib.residual_runtime import (  # noqa: E402
    batch_hash,
    configure_hf_offline,
    load_training_checkpoint,
    set_seed,
    sha256_file,
    write_json,
)
from experiments.lib.residual_scoring import (  # noqa: E402
    compute_block_first_order_scores,
    compute_block_taylor_scores,
    eligible_layers,
    transformer_layers,
)
from experiments.lib.residual_training import (  # noqa: E402
    build_optimizer,
    clone_model_state_to_cpu,
    partition_repeated_seed_batches,
    train_segment,
)

DEFAULT_WORKLOAD = "gpt2_medium_wikitext103"
DEFAULT_SEED = 42
DEFAULT_PRUNE_RATIO = 0.5
DEFAULT_RECOVERY_COUNT = 1
DEFAULT_MASK_COUNT = 32


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments/configs/paper_experiments.yaml",
    )
    parser.add_argument("--workload", default=DEFAULT_WORKLOAD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prune-ratio", type=float, default=DEFAULT_PRUNE_RATIO)
    parser.add_argument(
        "--k",
        "--K",
        "--recovery-count",
        dest="recovery_count",
        type=int,
        default=DEFAULT_RECOVERY_COUNT,
    )
    parser.add_argument("--mask-count", type=int, default=DEFAULT_MASK_COUNT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/diagnostics/damage_surrogate_validation.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not math.isfinite(args.prune_ratio) or not 0.0 < args.prune_ratio < 1.0:
        raise ValueError("prune-ratio must be finite and in (0, 1)")
    if (
        isinstance(args.recovery_count, bool)
        or not isinstance(args.recovery_count, int)
        or args.recovery_count != 1
    ):
        raise ValueError("damage surrogate validation currently requires fixed K=1")
    if (
        isinstance(args.mask_count, bool)
        or not isinstance(args.mask_count, int)
        or args.mask_count < 3
    ):
        raise ValueError("mask-count must be an integer of at least 3")


def _ensure_output_absent(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing damage validation output: {output}")


def _select_workload(args: argparse.Namespace):
    manifest = load_paper_manifest(args.manifest)
    jobs = plan_jobs(
        manifest.workloads,
        workload_names=[args.workload],
        prune_ratios=[args.prune_ratio],
        recovery_counts=[args.recovery_count],
        seeds=[args.seed],
    )
    if len(jobs) != 1:
        raise RuntimeError("Damage validation selection must produce exactly one job")
    return manifest, jobs[0]


def _dry_run_plan(args: argparse.Namespace, manifest, job) -> dict[str, object]:
    workload = job.workload
    return {
        "status": "dry_run",
        "manifest": str(manifest.path),
        "workload": workload.name,
        "model": workload.model,
        "dataset": workload.dataset,
        "checkpoint": str(workload.checkpoint),
        "data_dir": str(workload.data_dir),
        "seed": args.seed,
        "prune_ratio": args.prune_ratio,
        "recovery_count": args.recovery_count,
        "mask_count": args.mask_count,
        "random_mask_count": args.mask_count - 3,
        "score_families": ["magnitude", "first_order", "taylor"],
        "random_mask_allocation": "Taylor per-layer prune counts, generated from the run seed",
        "device": args.device,
        "output": str(args.output),
        "output_exists": args.output.exists(),
        "loads_model": False,
        "loads_gpu": False,
    }


def _source_digest(paths: Sequence[Path]) -> tuple[str, list[str]]:
    digest = hashlib.sha256()
    relative_paths: list[str] = []
    for path in sorted({Path(path).resolve() for path in paths}):
        try:
            relative = str(path.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            relative = str(path).replace("\\", "/")
        relative_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest(), relative_paths


def _mask_fingerprint(mask: Mapping[str, torch.Tensor], names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        value = mask[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _build_random_masks(
    layers: Sequence[Sequence[str]],
    shapes: Mapping[str, torch.Size],
    counts: Sequence[int],
    count: int,
    seed: int,
    occupied: set[str],
) -> list[MaskCandidate]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    names = [name for layer in layers for name in layer]
    candidates: list[MaskCandidate] = []
    attempts = 0
    while len(candidates) < count:
        attempts += 1
        if attempts > count * 20 + 20:
            raise RuntimeError("Could not generate distinct deterministic random masks")
        random_scores = {
            name: torch.rand(shapes[name], generator=generator, dtype=torch.float32)
            for name in names
        }
        mask = layer_masks(layers, random_scores, counts)
        fingerprint = _mask_fingerprint(mask, names)
        if fingerprint in occupied:
            continue
        occupied.add(fingerprint)
        candidates.append(
            MaskCandidate(
                id=f"random_{len(candidates):03d}",
                source="random",
                mask=mask,
                score_family="taylor",
            )
        )
    return candidates


def _build_candidates(
    layers: Sequence[Sequence[str]],
    delta: Mapping[str, torch.Tensor],
    magnitude_scores: Mapping[str, torch.Tensor],
    first_order_scores: Mapping[str, torch.Tensor],
    taylor_scores: Mapping[str, torch.Tensor],
    prune_ratio: float,
    mask_count: int,
    seed: int,
) -> tuple[list[MaskCandidate], dict[str, object]]:
    names = [name for layer in layers for name in layer]
    eligible_count = sum(delta[name].numel() for name in names)
    target_pruned = math.floor(prune_ratio * eligible_count)
    taylor_mask = global_mask(names, taylor_scores, target_pruned)
    taylor_counts = [
        sum(int((~taylor_mask[name]).sum().item()) for name in layer) for layer in layers
    ]
    candidates = [
        MaskCandidate(
            id="magnitude",
            source="magnitude",
            mask=global_mask(names, magnitude_scores, target_pruned),
            score_family="magnitude",
        ),
        MaskCandidate(
            id="first_order",
            source="first_order",
            mask=global_mask(names, first_order_scores, target_pruned),
            score_family="first_order",
        ),
        MaskCandidate(
            id="taylor",
            source="taylor",
            mask=taylor_mask,
            score_family="taylor",
        ),
    ]
    occupied = {_mask_fingerprint(candidate.mask, names) for candidate in candidates}
    candidates.extend(
        _build_random_masks(
            layers,
            {name: delta[name].shape for name in names},
            taylor_counts,
            mask_count - 3,
            seed,
            occupied,
        )
    )
    return candidates, {
        "eligible_parameter_count": eligible_count,
        "target_pruned": target_pruned,
        "per_layer_sizes": [sum(delta[name].numel() for name in layer) for layer in layers],
        "taylor_per_layer_prune_counts": taylor_counts,
        "mask_count": mask_count,
        "random_mask_count": mask_count - 3,
        "random_seed": seed,
        "exact_budget": True,
    }


def run(args: argparse.Namespace) -> Path:
    _validate_args(args)
    _ensure_output_absent(args.output)
    manifest, job = _select_workload(args)
    workload = job.workload
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
    configure_hf_offline()
    if not workload.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {workload.checkpoint}")

    checkpoint = load_training_checkpoint(workload.checkpoint)
    set_seed(0)
    train_loader, eval_loader, task_type = get_data_loaders(
        workload.model,
        workload.dataset,
        workload.batch_size,
        workload.seq_length,
        data_dir=str(workload.data_dir),
    )
    if task_type != workload.task_type:
        raise RuntimeError("Data loader task type does not match the manifest")
    training_pool = cache_batches(train_loader, workload.train_pool_batches, task_type)
    evaluation_batches = cache_batches(eval_loader, workload.eval_batches, task_type)
    if len(training_pool) != workload.train_pool_batches:
        raise RuntimeError("Training loader returned fewer batches than the manifest requires")
    if len(evaluation_batches) != workload.eval_batches:
        raise RuntimeError("Evaluation loader returned fewer batches than the manifest requires")
    batch_plan = partition_repeated_seed_batches(
        training_pool,
        workload.total_steps,
        args.recovery_count,
        workload.hvp_batches,
        args.seed,
    )

    model, model_family = load_model(
        workload.model,
        pretrained=False,
        checkpoint_path=None,
        device="cpu",
        dataset_name=workload.dataset,
    )
    model.load_state_dict(checkpoint.model_state, strict=True)
    model.to(args.device)
    scheduler_config = recovery_protocol(workload)["scheduler"]
    if not isinstance(scheduler_config, Mapping):
        raise RuntimeError("Invalid recovery scheduler protocol")
    optimizer = build_optimizer(
        model,
        checkpoint.optimizer_state,
        learning_rate=workload.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(scheduler_config["t_max"]),
        eta_min=float(scheduler_config["eta_min"]),
    )
    train_segment(
        model,
        optimizer,
        scheduler,
        batch_plan.training_segments[0],
        args.seed,
        args.device,
        task_type=task_type,
    )
    del optimizer, scheduler, train_loader, eval_loader
    current_state = clone_model_state_to_cpu(model)
    layers = eligible_layers(model, model_family)
    blocks = transformer_layers(model, model_family)
    names = [name for layer in layers for name in layer]
    delta = {
        name: current_state[name].detach().float() - checkpoint.model_state[name].detach().float()
        for name in names
    }
    magnitude_scores = {name: delta[name].abs() for name in names}
    first_order_scores, first_order_meta = compute_block_first_order_scores(
        model,
        batch_plan.scoring_batches[0],
        layers,
        delta,
        args.device,
        task_type=task_type,
        model_family=model_family,
        block_parameter_names=blocks,
    )
    taylor_scores, taylor_meta = compute_block_taylor_scores(
        model,
        batch_plan.scoring_batches[0],
        layers,
        delta,
        args.device,
        task_type=task_type,
        model_family=model_family,
        block_parameter_names=blocks,
    )
    candidates, mask_generation = _build_candidates(
        layers,
        delta,
        magnitude_scores,
        first_order_scores,
        taylor_scores,
        args.prune_ratio,
        args.mask_count,
        args.seed,
    )
    validation = validate_damage_surrogate(
        model,
        delta,
        layers,
        evaluation_batches,
        candidates,
        {
            "magnitude": magnitude_scores,
            "first_order": first_order_scores,
            "taylor": taylor_scores,
        },
        previous_reconstructed=checkpoint.model_state,
        device=args.device,
        task_type=task_type,
    )
    source_digest, source_files = _source_digest(
        [
            Path(__file__),
            ROOT / "experiments/lib/damage_surrogate_validation.py",
            ROOT / "experiments/lib/residual_scoring.py",
            ROOT / "experiments/lib/residual_masks.py",
            ROOT / "experiments/lib/residual_runtime.py",
            ROOT / "experiments/lib/residual_training.py",
            ROOT / "experiments/lib/paper_manifest.py",
        ]
    )
    output = {
        "schema_version": 1,
        "status": "complete",
        "manifest": str(manifest.path),
        "manifest_digest": sha256_file(manifest.path),
        "checkpoint": str(workload.checkpoint),
        "checkpoint_digest": sha256_file(workload.checkpoint),
        "source_digest": source_digest,
        "source_files": source_files,
        "batch_hash": batch_hash(evaluation_batches),
        "evaluation_batches": {
            "count": len(evaluation_batches),
            "sha256": batch_hash(evaluation_batches),
        },
        "device": str(args.device),
        "dtype": str(next(model.parameters()).dtype),
        "seed": args.seed,
        "workload": workload.name,
        "model": workload.model,
        "dataset": workload.dataset,
        "prune_ratio": args.prune_ratio,
        "recovery_count": args.recovery_count,
        "state_construction": {
            "reference_checkpoint_step": checkpoint.step,
            "pre_recovery_training_steps": len(batch_plan.training_segments[0]),
            "batch_plan": batch_plan.data_hashes(),
            "protocol": recovery_protocol(workload),
        },
        "mask_generation": mask_generation,
        "scoring": {
            "first_order": first_order_meta,
            "taylor": taylor_meta,
            "scoring_batch_hash": batch_hash(batch_plan.scoring_batches[0]),
        },
        "validation": validation,
    }
    write_json(args.output, output)
    return args.output


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    if not args.dry_run:
        _ensure_output_absent(args.output)
    manifest, job = _select_workload(args)
    if args.dry_run:
        print(json.dumps(_dry_run_plan(args, manifest, job), indent=2, sort_keys=True))
        return
    output = run(args)
    print(f"完成：{output}")


if __name__ == "__main__":
    main()
