"""New V100 cost and kappa confirmation using an existing fine-tuned checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def cases():
    plan = [("one_shot_mean", 1, 16, 0.0), ("one_shot_protected", 1, 16, 1.0)]
    plan += [(f"staged_kappa_{kappa:g}", 4, 4, kappa) for kappa in (0.0, 0.5, 1.0, 2.0)]
    for _, stages, batches, _ in plan:
        groups = [list(range(i * batches, (i + 1) * batches)) for i in range(stages)]
        assert stages * batches == 16
        assert sorted(index for group in groups for index in group) == list(range(16))
    return plan


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparsity", type=float, default=0.98)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    plan = cases()
    if args.plan_only:
        print(json.dumps({"backward_budget": 16, "unique_scoring_batches": 16, "cases": plan}))
        return
    if args.checkpoint is None or args.data_root is None or args.out_dir is None:
        parser.error("--checkpoint, --data-root and --out-dir are required")
    if not 0.0 < args.sparsity < 1.0:
        parser.error("--sparsity must be between zero and one")

    import torch
    from run_residual_ablation_small import geometric_schedule, grad_moments, run_iterative
    from run_residual_sweep_small import REGIMES, load_regime, _jsonable
    from experiments.lib.residual_runtime import batch_hash, set_seed, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches

    checkpoint_hash = sha256(args.checkpoint)
    script_hash = sha256(Path(__file__))
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    set_seed(args.seed)
    context = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    sweep = context["sweep"]
    scoring = partition_repeated_seed_batches(context["pool"], 100, 1, 16, args.seed).scoring_batches[0]
    scoring_hash = batch_hash(scoring)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("SETUP " + json.dumps({"checkpoint_sha256": checkpoint_hash,
                                 "scoring_sha256": scoring_hash, "uncompressed": context["uncompressed"]}), flush=True)
    for name, stage_count, batch_count, kappa in plan:
        path = args.out_dir / f"{name}_p{args.sparsity}_seed{args.seed}.json"
        identity = {"checkpoint_sha256": checkpoint_hash, "scoring_sha256": scoring_hash,
                    "seed": args.seed, "sparsity": args.sparsity, "stages": stage_count,
                    "batches_per_stage": batch_count, "kappa": kappa, "script_sha256": script_hash}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("identity") != identity or existing.get("status") != "completed":
                raise RuntimeError(f"Existing result has a different identity: {path}")
            print("RESULT " + str(path), flush=True)
            continue
        stage_index = 0
        set_seed(args.seed)
        sweep.apply(None)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def scorer(current, unused_batches, keep_prev, seed):
            nonlocal stage_index
            start = stage_index * batch_count
            selected = scoring[start:start + batch_count]
            if len(selected) != batch_count:
                raise RuntimeError("Stage scoring data exhausted")
            stage_index += 1
            mean, m2, losses = grad_moments(current, selected, kappa > 0.0)
            values = {}
            for i, parameter in enumerate(current.names):
                value = mean[i].abs()
                if m2 is not None:
                    value = value + kappa * (m2[i] / (batch_count - 1)).clamp_min(0).sqrt() / math.sqrt(batch_count)
                values[parameter] = value
            return values, {"score_losses": losses}

        schedule = geometric_schedule(args.sparsity, stage_count)
        evaluation, stage_records = run_iterative(sweep, scoring, schedule, scorer, args.seed,
                                                  eval_at=[args.sparsity])
        value = evaluation[str(args.sparsity)]
        if not math.isfinite(value["val"]):
            raise RuntimeError("Nonfinite evaluation loss")
        record = {"status": "completed", "purpose": "new V100 fixed-budget cost and kappa confirmation",
                  "identity": identity, "model": "pythia-410m", "dataset": "alpaca",
                  "reference": "pretrained_base", "checkpoint_path": str(args.checkpoint),
                  "backward_budget": stage_count * batch_count, "unique_scoring_batches": 16,
                  "stage_scoring_policy": "disjoint consecutive groups from the same 16-batch pool",
                  "uncompressed": context["uncompressed"], "evaluation": value,
                  "delta_loss": value["val"] - context["uncompressed"]["val"],
                  "stage_records": stage_records,
                  "score_seconds": sum(item["score_s"] for item in stage_records),
                  "selection_seconds": sum(item["select_s"] for item in stage_records),
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                  "evaluation_sha256": batch_hash(context["eval_batches"]),
                  "source_provenance": context["provenance"]}
        write_json(path, _jsonable(record))
        print("RESULT " + str(path), flush=True)
    sweep.apply(None)


if __name__ == "__main__":
    main()
