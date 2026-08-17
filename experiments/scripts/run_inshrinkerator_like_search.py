"""
Inshrinkerator-like per-layer-type pruning ratio search.

Searches for the best per-type (attn/mlp/...) pruning ratios under a
quality constraint, using magnitude and sensitivity metrics.

Usage:
    python experiments/scripts/run_inshrinkerator_like_search.py \
        --model gpt2-small --dataset wikitext103 \
        --checkpoint checkpoints/gpt2_small_checkpoint.pt \
        --epsilons 0.01,0.05 \
        --num_steps 50 --eval_batches 20 \
        --device cuda

    python experiments/scripts/run_inshrinkerator_like_search.py \
        --model bert-large --dataset sst2 \
        --checkpoint checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000_final.pt \
        --epsilons 0.01,0.05 \
        --device cuda
"""

import argparse
import copy
import json
import math
import os
import re
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from baselines.inshrinkerator.per_type_search import SearchConfig, search_best_config
from dacp.pruning.param_schema import build_type_map
from experiments.lib.data import cache_batches, get_data_loaders
from experiments.lib.evaluation import compute_quality_drop, evaluate
from experiments.lib.importance_compare.scoring import compute_scores_by_method
from experiments.lib.models import load_model
from experiments.lib.residual_runtime import batch_hash, set_seed, sha256_file
from experiments.scripts.run_paper_experiments import _source_provenance


def _get_model_family(model_name: str) -> str:
    if "gpt2" in model_name:
        return "gpt2"
    elif "bert" in model_name:
        return "bert"
    elif "resnet" in model_name:
        return "resnet"
    return "gpt2"


def _stable_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    if not slug:
        raise ValueError(f"Cannot build a stable slug from {value!r}")
    return slug


def _result_name(model: str, dataset: str, seed: int, epsilon: float, count: int) -> str:
    prefix = f"{_stable_slug(model)}_{_stable_slug(dataset)}_seed{seed}"
    if count == 1:
        return f"{prefix}.json"
    epsilon_slug = f"{epsilon:.8g}".replace("-", "m").replace(".", "p")
    return f"{prefix}_eps{epsilon_slug}.json"


def _build_search_payload(
    result,
    *,
    model: str,
    dataset: str,
    seed: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    source_digest: str,
    scoring_batches,
    evaluation_batches,
    protect_fraction: float,
) -> dict[str, object]:
    payload = asdict(result)
    payload.update(
        {
            "schema_version": 1,
            "status": "complete",
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "source_digest": source_digest,
            "scoring_batch_sha256": batch_hash(scoring_batches),
            "evaluation_batch_sha256": batch_hash(evaluation_batches),
            "scoring_batch_count": len(scoring_batches),
            "evaluation_batch_count": len(evaluation_batches),
            "protect_fraction": protect_fraction,
            "quality_drop": float(result.quality_drop_pct),
        }
    )
    return payload


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing search result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Inshrinkerator-like per-layer-type pruning search"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default=None)
    parser.add_argument("--epsilons", type=str, default="0.01,0.05")
    parser.add_argument("--coarse_candidates", type=str, default="0.0,0.2,0.4")
    parser.add_argument("--fine_step", type=float, default=0.1)
    parser.add_argument("--fine_radius", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--hvp_batches", type=int, default=4)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protect-fraction",
        "--protect_fraction",
        dest="protect_fraction",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="results/paper_results/inshrinkerator_search",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    epsilons = [float(e) for e in args.epsilons.split(",")]
    coarse = [float(c) for c in args.coarse_candidates.split(",")]
    model_family = _get_model_family(args.model)

    if not epsilons or any(not math.isfinite(value) for value in epsilons):
        raise ValueError("--epsilons must contain finite values")
    if len(set(epsilons)) != len(epsilons):
        raise ValueError("--epsilons must not contain duplicates")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if not 0.0 <= args.protect_fraction < 1.0:
        raise ValueError("--protect-fraction must be finite in [0, 1)")
    if args.hvp_batches < 1 or args.num_steps < args.hvp_batches:
        raise ValueError("--num_steps must be at least --hvp_batches")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "dataset": args.dataset,
                    "model_family": model_family,
                    "seed": args.seed,
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "hvp_batches": args.hvp_batches,
                    "eval_batches": args.eval_batches,
                    "protect_fraction": args.protect_fraction,
                    "epsilons": epsilons,
                    "output_dir": str(Path(args.output_dir).resolve()),
                },
                indent=2,
            )
        )
        return

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    source_digest = _source_provenance(Path(__file__).resolve())["source_state_sha256"]
    if not isinstance(source_digest, str):
        raise RuntimeError("Source provenance did not return a digest")
    output_paths = [
        Path(args.output_dir)
        / _result_name(args.model, args.dataset, args.seed, epsilon, len(epsilons))
        for epsilon in epsilons
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing search results: {existing}")
    set_seed(args.seed)

    print("=" * 70)
    print("Inshrinkerator-like Per-Layer-Type Pruning Search")
    print("=" * 70)
    print(f"Model: {args.model} | Dataset: {args.dataset}")
    print(f"Epsilons: {epsilons}")
    print(f"Coarse candidates: {coarse}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # 1. Load model
    print("\n[1/5] Loading model...")
    model, model_family = load_model(
        args.model,
        pretrained=True,
        checkpoint_path=str(checkpoint_path),
        device="cpu",
        dataset_name=args.dataset,
    )

    # 2. Load data
    print(f"[2/5] Loading data ({args.dataset})...")
    train_loader, val_loader, task_type = get_data_loaders(
        args.model,
        args.dataset,
        args.batch_size,
        args.seq_length,
        data_dir=args.data_dir or "./data",
    )
    cached_train = cache_batches(train_loader, args.num_steps, task_type)
    cached_eval = cache_batches(val_loader, args.eval_batches, task_type)

    # 3. Compute importance scores (magnitude + first-order)
    print("[3/5] Computing importance scores...")
    model_for_scoring = copy.deepcopy(model).to(args.device)
    scores_by_metric = compute_scores_by_method(
        model_for_scoring,
        cached_train,
        task_type,
        methods=["magnitude", "first-order"],
        alpha=args.alpha,
        hvp_batches=args.hvp_batches,
        model_family=model_family,
    )
    del model_for_scoring
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Build type map
    print("[4/5] Building type map...")
    weights = {n: p.detach().cpu() for n, p in model.named_parameters()}
    type_map = build_type_map(weights, model_family)

    # Print type distribution
    from collections import Counter

    type_counts = Counter(type_map.values())
    for lt, cnt in sorted(type_counts.items()):
        print(f"  {lt}: {cnt} params")

    # 5. Search
    print("[5/5] Running search...")
    search_cfg = SearchConfig(
        coarse_candidates=coarse,
        fine_step=args.fine_step,
        fine_radius=args.fine_radius,
        metrics=["magnitude", "sensitivity"],
        epsilons=epsilons,
    )

    def model_factory():
        return copy.deepcopy(model)

    for index, eps in enumerate(epsilons):
        print(f"\n--- epsilon = {eps} ---")
        result = search_best_config(
            model_factory=model_factory,
            scores_by_metric=scores_by_metric,
            type_map=type_map,
            cached_eval=cached_eval,
            task_type=task_type,
            device=args.device,
            evaluate_fn=evaluate,
            quality_drop_fn=compute_quality_drop,
            epsilon=eps,
            config=search_cfg,
            model_family=model_family,
        )

        print(f"  Best metric: {result.best_metric}")
        print(f"  Per-type ratios: {result.per_type_ratios}")
        print(f"  Actual global ratio: {result.actual_global_ratio:.4f}")
        print(f"  Quality drop: {result.quality_drop_pct:.2f}%")
        print(f"  Baseline: {result.baseline_metrics}")
        print(f"  Pruned:   {result.pruned_metrics}")

        payload = _build_search_payload(
            result,
            model=args.model,
            dataset=args.dataset,
            seed=args.seed,
            checkpoint=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            source_digest=source_digest,
            scoring_batches=cached_train[: args.hvp_batches],
            evaluation_batches=cached_eval,
            protect_fraction=args.protect_fraction,
        )
        path = output_paths[index]
        _write_new_json(path, payload)
        print(f"  Saved: {path}")

    print("\nSearch complete!")


if __name__ == "__main__":
    main()
