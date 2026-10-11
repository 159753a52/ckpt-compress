"""Find a measured low-damage operating point, using a separate calibration set.

Try the paper's larger K16/B16 budget at 98% first, then less aggressive
sparsities with the inexpensive disjoint K4/B4 selector. Only seed 42 is used.
The prior K4/B4 98% stress result is reused, rather than recomputed.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--gate-ppl-tolerance", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not 0 < args.gate_ppl_tolerance < 1:
        ap.error("invalid calibration tolerance")

    import torch
    from run_residual_ablation_small import geometric_schedule, grad_moments, run_iterative, score_ucb
    from run_residual_sweep_small import REGIMES, load_regime, _jsonable
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, set_seed, sha256_file, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches

    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    context = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    sweep, pool = context["sweep"], context["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, args.seed)
    scoring = plan.scoring_batches[0]
    offset = len(plan.training_segments[0])
    scoring_indices = set(plan.selected_pool_indices[offset:offset + 16])
    gate_indices = [i for i in range(len(pool)) if i not in scoring_indices][:64]
    gate = [pool[i] for i in gate_indices]
    assert len(scoring_indices) == 16 and not scoring_indices.intersection(gate_indices)
    sweep.apply(None)
    baseline_gate = evaluate_lm(sweep.model, gate, "cuda:0")
    baseline_test = context["uncompressed"]
    common = {"checkpoint_sha256": context["provenance"]["checkpoint_sha256"],
              "scoring_sha256": batch_hash(scoring), "gate_sha256": batch_hash(gate),
              "heldout_sha256": batch_hash(context["eval_batches"]), "seed": args.seed,
              "gate_ppl_tolerance": args.gate_ppl_tolerance,
              "script_sha256": sha256_file(Path(__file__))}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(args.out_dir / "run-manifest.json", {"identity": common, "code_commit": commit,
               "source_provenance": context["provenance"], "gate_pool_indices": gate_indices,
               "scoring_pool_indices": sorted(scoring_indices), "baseline_gate": baseline_gate,
               "baseline_heldout": baseline_test, "gpu": torch.cuda.get_device_name(0)})
    cases = [(0.98, 16, 16)] + [(p, 4, 4) for p in (0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.30, 0.10)]
    for sparsity, stage_count, batches in cases:
        identity = {**common, "sparsity": sparsity, "stages": stage_count, "batches_per_stage": batches,
                    "scoring_policy": "reuse16" if stage_count == 16 else "disjoint4x4"}
        path = args.out_dir / f"staged_K{stage_count}_B{batches}_p{sparsity}_seed{args.seed}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result.get("identity") != identity or result.get("status") != "completed":
                raise RuntimeError("existing calibration identity conflict")
            print("SKIP " + str(path), flush=True)
        else:
            set_seed(args.seed)
            sweep.apply(None)
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            if stage_count == 16:
                scorer = score_ucb
            else:
                stage = 0

                def scorer(current, unused, keep_prev, seed):
                    nonlocal stage
                    selected = scoring[stage * 4:(stage + 1) * 4]
                    stage += 1
                    mean, m2, losses = grad_moments(current, selected, True)
                    scores = {name: mean[i].abs() + (m2[i] / 3).clamp_min(0).sqrt() / 2
                              for i, name in enumerate(current.names)}
                    return scores, {"score_losses": losses}

            _, stages, keep = run_iterative(sweep, scoring, geometric_schedule(sparsity, stage_count),
                                            scorer, args.seed, eval_at=[], return_keep=True)
            score_seconds = time.perf_counter() - began
            sweep.apply(keep)
            gate_result = evaluate_lm(sweep.model, gate, "cuda:0")
            gate_change = gate_result["loss"] - baseline_gate["loss"]
            passed = math.isfinite(gate_change) and gate_change <= math.log1p(args.gate_ppl_tolerance)
            heldout = sweep.evaluate() if passed else None
            result = {"status": "completed", "purpose": "low-damage operating-point calibration",
                      "identity": identity, "code_commit": commit, "reference": "pretrained_base",
                      "baseline_gate": baseline_gate, "baseline_heldout": baseline_test,
                      "calibration": gate_result, "gate_delta_loss": gate_change,
                      "gate_relative_ppl_change": math.expm1(gate_change), "passed_gate": passed,
                      "heldout": heldout,
                      "heldout_delta_loss": heldout["val"] - baseline_test["val"] if heldout else None,
                      "backward_budget": stage_count * batches, "unique_scoring_batches": 16,
                      "score_seconds": score_seconds, "stage_records": stages,
                      "kept": sum(int(mask.sum()) for mask in keep.values()),
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "gpu": torch.cuda.get_device_name(0)}
            write_json(path, _jsonable(result))
            print(f"RESULT {path} gate_ppl={result['gate_relative_ppl_change']:.6f} passed={passed}", flush=True)
            del keep
            sweep.apply(None)
            sweep.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        if result["passed_gate"]:
            write_json(args.out_dir / "selected.json", result)
            print("SELECTED " + str(path), flush=True)
            break
    else:
        raise RuntimeError("No tested operating point meets the calibration tolerance")


if __name__ == "__main__":
    main()
