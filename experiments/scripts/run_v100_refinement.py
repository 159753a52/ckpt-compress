"""New scoring-budget and penalty calibration at fixed residual sparsity.

Select on training calibration only; evaluate the selected mask on validation
once. Reuse the already measured K4/B4/kappa1 point and its saved mask.
"""
from pathlib import Path
import argparse
import json
import math
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--legacy-mask-record", type=Path, required=True)
    ap.add_argument("--sparsity", type=float, default=0.7)
    args = ap.parse_args()
    import torch
    from run_residual_sweep_small import REGIMES, load_regime
    from run_residual_ablation_small import geometric_schedule, grad_moments, run_iterative
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, set_seed, sha256_file, write_json
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    sweep, pool = ctx["sweep"], ctx["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    initial = plan.scoring_batches[0]
    used = plan.selected_pool_indices[len(plan.training_segments[0]):][:16]
    gate_indices = [i for i in range(len(pool)) if i not in set(used)][:64]
    extra = [i for i in range(len(pool)) if i not in set(used + gate_indices)]
    random.Random(42).shuffle(extra)
    scoring = initial + [pool[i] for i in extra[:48]]
    gate = [pool[i] for i in gate_indices]
    assert len(scoring) == 64 and not set(used + extra[:48]).intersection(gate_indices)
    legacy = json.loads(args.legacy_mask_record.read_text())
    if legacy["identity"]["checkpoint_sha256"] != ctx["provenance"]["checkpoint_sha256"] or legacy["identity"]["gate_sha256"] != batch_hash(gate):
        raise RuntimeError("legacy input identity differs")
    baseline_gate = legacy["calibration_reused"]["baseline_gate"] if "baseline_gate" in legacy["calibration_reused"] else None
    # The validation record preserves the gate measurement, not its baseline.
    old = json.loads((ROOT / "results/v100_quality_calibration_20261011/staged_K4_B4_p0.7_seed42.json").read_text())
    baseline_gate = old["baseline_gate"]
    if old["identity"]["gate_sha256"] != batch_hash(gate):
        raise RuntimeError("calibration baseline input differs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    common = {"seed": 42, "checkpoint_sha256": ctx["provenance"]["checkpoint_sha256"],
              "gate_sha256": batch_hash(gate), "heldout_sha256": batch_hash(ctx["eval_batches"]),
              "sparsity": args.sparsity, "script_sha256": sha256_file(Path(__file__))}
    configs = [(4, 4, k) for k in (0.0, 0.5, 2.0)]
    configs += [(4, b, k) for b in (8, 16) for k in (0.0, 0.5, 1.0, 2.0)]
    configs += [(8, 4, 1.0), (16, 4, 1.0)]
    write_json(args.out_dir / "run-manifest.json", {"identity": common, "configs": configs,
               "gate_pool_indices": gate_indices, "scoring_pool_indices": used + extra[:48],
               "baseline_gate": baseline_gate, "baseline_heldout": ctx["uncompressed"],
               "code_commit": commit, "gpu": torch.cuda.get_device_name(0)})
    best = {"gate_delta_loss": old["gate_delta_loss"], "reused": True,
            "config": [4, 4, 1.0], "mask_path": str(args.legacy_mask_record.parent / legacy["mask"]["file"])}
    records = []
    for stages, batches, penalty in configs:
        tag = f"K{stages}_B{batches}_k{penalty:g}_p{args.sparsity}"
        path = args.out_dir / f"{tag}.json"
        identity = {**common, "config": [stages, batches, penalty], "scoring_sha256": batch_hash(scoring[:stages * batches])}
        if path.exists():
            rec = json.loads(path.read_text())
            if rec["identity"] != identity or rec["status"] != "completed":
                raise RuntimeError("existing result identity conflict")
            print("SKIP " + tag, flush=True)
        else:
            set_seed(42)
            sweep.apply(None)
            stage = 0
            def scorer(current, unused, keep_prev, seed):
                nonlocal stage
                bs = scoring[stage * batches:(stage + 1) * batches]
                stage += 1
                mean, m2, losses = grad_moments(current, bs, penalty > 0)
                scores = {n: mean[i].abs() + (penalty * (m2[i] / (batches - 1)).clamp_min(0).sqrt() / math.sqrt(batches) if penalty else 0)
                          for i, n in enumerate(current.names)}
                return scores, {"score_losses": losses}
            began = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            _, detail, keep = run_iterative(sweep, scoring, geometric_schedule(args.sparsity, stages), scorer, 42, eval_at=[], return_keep=True)
            score_seconds = time.perf_counter() - began
            sweep.apply(keep)
            result = evaluate_lm(sweep.model, gate, "cuda:0")
            delta = result["loss"] - baseline_gate["loss"]
            rec = {"status": "completed", "identity": identity, "config": [stages, batches, penalty],
                   "gate": result, "gate_delta_loss": delta, "gate_ppl_change_percent": 100 * math.expm1(delta),
                   "stage_records": detail, "backward_budget": stages * batches,
                   "score_seconds": score_seconds, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "code_commit": commit}
            if delta < best["gate_delta_loss"]:
                mask = args.out_dir / f"{tag}-mask.pt"
                torch.save({n: x.cpu() for n, x in keep.items()}, mask)
                rec["mask_path"] = str(mask)
                rec["mask_sha256"] = sha256_file(mask)
            write_json(path, rec)
            del keep
            sweep.apply(None)
            sweep.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print("RESULT " + tag + " " + str(rec["gate_ppl_change_percent"]), flush=True)
        records.append(rec)
        if rec["gate_delta_loss"] < best["gate_delta_loss"]:
            best = rec
    selected_path = args.out_dir / "selected.json"
    if selected_path.exists():
        print("SKIP selected validation", flush=True)
        return
    keep = torch.load(best["mask_path"], map_location="cpu", weights_only=True)
    sweep.apply(keep)
    heldout = sweep.evaluate() if not best.get("reused") else legacy["evaluation"]
    write_json(selected_path, {**best, "identity_common": common, "baseline": ctx["uncompressed"],
               "evaluation": heldout, "validation_ppl_change_percent": 100 * math.expm1(heldout["val"] - ctx["uncompressed"]["val"]),
               "selection_rule": "minimum training-calibration loss at fixed sparsity; heldout validation not used for selection",
               "legacy_validation_reused": best.get("reused", False)})
    print("SELECTED " + str(best["config"]), flush=True)


if __name__ == "__main__":
    main()
