"""Output-category probe prefixes: coverage, exact-GN error and restoration loss.

One 64-round run supplies prefixes 1,2,4,8,16,32,64 without repeating backprop.
The selected point uses training calibration; validation is never a selector.
"""
from pathlib import Path
import argparse
import json
import math
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
    ap.add_argument("--estimator", choices=("topk", "orthogonal"), default="topk")
    args = ap.parse_args()
    import torch
    from run_residual_sweep_small import REGIMES, load_regime, global_keep_masks
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, set_seed, sha256_file, write_json
    from experiments.lib.residual_probe_scoring import compute_probe_gn_scores
    from experiments.lib.residual_probe_reference import exact_gn_product
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "completed.json").exists():
        print("SKIP completed output-probe experiment", flush=True)
        return
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    s, pool = ctx["sweep"], ctx["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    scoring = plan.scoring_batches[0][:4]
    offset = len(plan.training_segments[0])
    used = set(plan.selected_pool_indices[offset:offset + 16])
    gate_indices = [i for i in range(len(pool)) if i not in used][:64]
    gate = [pool[i] for i in gate_indices]
    assert not used.intersection(gate_indices)
    s.apply(None)
    baseline_gate = evaluate_lm(s.model, gate, "cuda:0")
    common = {"seed": 42, "checkpoint_sha256": ctx["provenance"]["checkpoint_sha256"],
              "scoring_sha256": batch_hash(scoring), "gate_sha256": batch_hash(gate),
              "heldout_sha256": batch_hash(ctx["eval_batches"]), "scoring_batches": 4,
              "script_sha256": sha256_file(Path(__file__)), "probe_library_sha256": sha256_file(ROOT / "experiments/lib/residual_probe_scoring.py"), "estimator": args.estimator}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(args.out_dir / "run-manifest.json", {"identity": common, "code_commit": commit,
               "baseline_gate": baseline_gate, "baseline_heldout": ctx["uncompressed"], "gpu": torch.cuda.get_device_name(0)})
    started = time.perf_counter()
    exact = exact_gn_product(s.model, scoring, s.names, s.delta, "cuda:0")
    exact_seconds = time.perf_counter() - started
    denominator = sum(float(x.double().square().sum()) for x in exact.values())
    captures = []
    def prefix(rounds, scores, curvature, metadata):
        error = sum(float((curvature[n].double() - exact[n].double()).square().sum()) for n in s.names)
        rel = math.sqrt(error / max(denominator, 1e-300))
        masks = {p: {n: m.cpu() for n, m in keep.items()} for p, keep, seconds in global_keep_masks(scores, s.names, [.5, .7, .8, .9], "cuda:0")}
        captures.append((rounds, masks, metadata, rel))
        print(f"PREFIX m={rounds} GN_error={rel:.6f}", flush=True)
    set_seed(42)
    torch.cuda.reset_peak_memory_stats()
    if args.estimator == "topk":
        score, meta = compute_probe_gn_scores(s.model, scoring, s.layers, s.delta, "cuda:0",
                         estimator="topk", rounds=64, seed=42, model_family=s.family,
                         block_parameter_names=s.blocks, report_rounds=(1, 2, 4, 8, 16, 32, 64), on_prefix=prefix)
    else:
        from experiments.lib.residual_orthogonal_probes import orthogonal_scores
        score, meta = orthogonal_scores(s.model, scoring, s.names, s.delta, "cuda:0", 64,
                         report_rounds=(1, 2, 4, 8, 16, 32, 64), on_prefix=prefix)
    del score, exact
    peak = torch.cuda.max_memory_allocated() / 2**30
    best = None
    for rounds, masks, coverage, error in captures:
        points = {}
        for p, keep in masks.items():
            s.apply(keep)
            gate_result = evaluate_lm(s.model, gate, "cuda:0")
            heldout = s.evaluate()
            points[str(p)] = {"gate": gate_result, "evaluation": heldout,
                             "gate_ppl_change_percent": 100 * math.expm1(gate_result["loss"] - baseline_gate["loss"]),
                             "heldout_ppl_change_percent": 100 * math.expm1(heldout["val"] - ctx["uncompressed"]["val"])}
            if p == .7 and (best is None or gate_result["loss"] < best["gate"]["loss"]):
                mask = args.out_dir / f"{args.estimator}_m{rounds}_p0.7-mask.pt"
                torch.save(keep, mask)
                best = {"rounds": rounds, "gate": gate_result, "mask_path": str(mask), "mask_sha256": sha256_file(mask)}
        rec = {"status": "completed", "identity": common, "code_commit": commit, "rounds": rounds,
               "coverage": coverage, "relative_l2_error_full_gn_product": error, "points": points,
               "exact_gn_seconds": exact_seconds, "peak_allocated_gib_including_prefix_masks": peak,
               "prefixes_share_gradient_and_backprop": True}
        write_json(args.out_dir / f"{args.estimator}_m{rounds}_seed42.json", rec)
        print("RESULT m=" + str(rounds) + " " + str(points["0.7"]["heldout_ppl_change_percent"]), flush=True)
        masks.clear()
    write_json(args.out_dir / "selected.json", {**best, "identity": common,
               "selection_rule": "minimum training-calibration loss at fixed 70% sparsity; heldout not used to select"})
    write_json(args.out_dir / "completed.json", {"status": "completed", "identity": common, "code_commit": commit,
               "probe": meta.get("probe", meta), "total_seconds_including_selection": meta.get("total_seconds"),
               "selection": best, "experiment": "output probability factors and GN estimation; coverage is not confidence"})


if __name__ == "__main__":
    main()
