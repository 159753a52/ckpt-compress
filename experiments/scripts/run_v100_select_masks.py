"""Select residual keep masks at one deletion rate for DACP-G, ExCP and Inshrinkerator.

DACP-G uses staged first-order scoring with the variance penalty; its stage,
batch and penalty configuration is chosen on the training-calibration (gate)
loss only. ExCP and Inshrinkerator use their fixed rules (residual magnitude;
block weight-gradient product on eight scoring batches). Held-out validation is
evaluated once per selected mask and is never used for selection.
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

GRID = [(4, 4, k, False) for k in (0.0, 0.5, 1.0, 2.0)]
GRID += [(4, b, k, False) for b in (8, 16) for k in (0.0, 0.5, 1.0, 2.0)]
GRID += [(8, 4, 1.0, False), (8, 8, 1.0, False), (8, 8, 2.0, False), (16, 4, 1.0, False), (16, 4, 2.0, False)]
# Shared batches: every stage rescores the same B batches (the 16x16 DACP-G setting of the tolerance and 98% experiments).
SHARED = [(k, b, kappa, True) for k, b in ((8, 16), (16, 8), (16, 16)) for kappa in (1.0, 2.0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sparsity", type=float, required=True)
    ap.add_argument("--grid", choices=("disjoint", "shared", "all"), default="all")
    args = ap.parse_args()
    import torch
    from run_residual_sweep_small import REGIMES, load_regime, global_keep_masks
    from run_residual_ablation_small import geometric_schedule, grad_moments, run_iterative
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, set_seed, sha256_file, write_json
    from experiments.lib.residual_scoring import compute_block_first_order_scores

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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sweep.apply(None)
    baseline_gate = evaluate_lm(sweep.model, gate, "cuda:0")
    common = {"seed": 42, "checkpoint_sha256": ctx["provenance"]["checkpoint_sha256"],
              "gate_sha256": batch_hash(gate), "heldout_sha256": batch_hash(ctx["eval_batches"]),
              "sparsity": args.sparsity, "script_sha256": sha256_file(Path(__file__))}
    write_json(args.out_dir / "run-manifest.json", {"identity": common, "dacp_grid": {"disjoint": GRID, "shared": SHARED, "all": GRID + SHARED}[args.grid],
               "gate_pool_indices": gate_indices, "scoring_pool_indices": used + extra[:48],
               "baseline_gate": baseline_gate, "baseline_heldout": ctx["uncompressed"],
               "code_commit": commit, "gpu": torch.cuda.get_device_name(0)})

    def finalize(name, keep, fields):
        mask = args.out_dir / f"{name}-mask.pt"
        torch.save({n: x.cpu() for n, x in keep.items()}, mask)
        sweep.apply(keep)
        gate_eval = evaluate_lm(sweep.model, gate, "cuda:0")
        heldout = sweep.evaluate()
        sweep.apply(None)
        rec = {**fields, "status": "completed", "identity_common": common,
               "mask_path": str(mask.resolve()), "mask_sha256": sha256_file(mask),
               "gate": gate_eval, "gate_ppl_change_percent": 100 * math.expm1(gate_eval["loss"] - baseline_gate["loss"]),
               "baseline_gate": baseline_gate, "baseline": ctx["uncompressed"], "evaluation": heldout,
               "validation_ppl_change_percent": 100 * math.expm1(heldout["val"] - ctx["uncompressed"]["val"]),
               "code_commit": commit}
        write_json(args.out_dir / f"{name}.json", rec)
        print("SELECTED " + name + " " + json.dumps({k: rec[k] for k in ("gate_ppl_change_percent", "validation_ppl_change_percent")}), flush=True)

    # DACP-G: staged first-order scores with the variance penalty, configuration chosen on the gate loss.
    candidates, best, best_keep = [], None, None
    grid = {"disjoint": GRID, "shared": SHARED, "all": GRID + SHARED}[args.grid]
    for stages, batches, penalty, shared in grid:
        tag = f"K{stages}_B{batches}_k{penalty:g}" + ("_shared" if shared else "")
        set_seed(42)
        sweep.apply(None)
        stage = 0

        def scorer(current, unused, keep_prev, seed):
            nonlocal stage
            bs = scoring[:batches] if shared else scoring[stage * batches:(stage + 1) * batches]
            stage += 1
            mean, m2, losses = grad_moments(current, bs, penalty > 0)
            scores = {n: mean[i].abs() + (penalty * (m2[i] / (batches - 1)).clamp_min(0).sqrt() / math.sqrt(batches) if penalty else 0)
                      for i, n in enumerate(current.names)}
            return scores, {"score_losses": losses}

        began = time.perf_counter()
        _, detail, keep = run_iterative(sweep, scoring, geometric_schedule(args.sparsity, stages), scorer, 42, eval_at=[], return_keep=True)
        seconds = time.perf_counter() - began
        sweep.apply(keep)
        gate_eval = evaluate_lm(sweep.model, gate, "cuda:0")
        delta = gate_eval["loss"] - baseline_gate["loss"]
        rec = {"tag": tag, "config": [stages, batches, penalty] + (["shared"] if shared else []), "backward_budget": stages * batches,
               "scoring_sha256": batch_hash(scoring[:batches] if shared else scoring[:stages * batches]), "gate": gate_eval,
               "gate_delta_loss": delta, "gate_ppl_change_percent": 100 * math.expm1(delta), "score_seconds": seconds}
        candidates.append(rec)
        write_json(args.out_dir / "dacp_candidates.json", candidates)
        print(f"RESULT dacp {tag} gate_ppl={rec['gate_ppl_change_percent']:.4f} seconds={seconds:.1f}", flush=True)
        if best is None or delta < best["gate_delta_loss"]:
            best, best_keep = rec, {n: x.cpu() for n, x in keep.items()}
        del keep
        sweep.apply(None)
        sweep.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    finalize("dacp", best_keep, {"method": "iterative_ucb_disjoint", "config": best["config"],
             "backward_budget": best["backward_budget"],
             "selection_rule": "minimum training-calibration loss over the registered grid; heldout validation not used"})
    del best_keep

    # ExCP: residual magnitude, exact global threshold.
    sweep.apply(None)
    scores = {n: sweep.delta[n].abs() for n in sweep.names}
    (_, keep, _), = list(global_keep_masks(scores, sweep.names, [args.sparsity], sweep.device))
    del scores
    finalize("excp", keep, {"method": "excp_style", "config": ["residual-magnitude"], "backward_budget": 0})
    del keep

    # Inshrinkerator: block weight-gradient product on the first eight scoring batches (as in the 70% runs).
    sweep.apply(None)
    scores, _ = compute_block_first_order_scores(sweep.model, initial[:8], sweep.layers, sweep.cur_cpu, sweep.device,
                                                 model_family=sweep.family, block_parameter_names=sweep.blocks)
    (_, keep, _), = list(global_keep_masks(scores, sweep.names, [args.sparsity], sweep.device))
    del scores
    finalize("ins", keep, {"method": "inshrinkerator_style", "config": ["block-weight-gradient"], "backward_budget": 8})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
