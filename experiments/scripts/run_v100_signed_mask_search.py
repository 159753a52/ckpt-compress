"""Screen signed residual damage and quadratic damping before physical encoding."""
from pathlib import Path
import argparse
import json
import math
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from run_residual_sweep_small import REGIMES, load_regime, global_keep_masks
    from run_residual_ablation_small import grad_moments
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, sha256_file, write_json
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    s, pool = ctx["sweep"], ctx["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    offset = len(plan.training_segments[0])
    used = plan.selected_pool_indices[offset:offset + 16]
    gate_indices = [i for i in range(len(pool)) if i not in set(used)][:64]
    extra = [i for i in range(len(pool)) if i not in set(used + gate_indices)]
    random.Random(42).shuffle(extra)
    scoring_indices = used + extra[:48]
    assert len(scoring_indices) == 64 and not set(scoring_indices).intersection(gate_indices)
    scoring = [pool[i] for i in scoring_indices]
    gate = [pool[i] for i in gate_indices]
    baseline = evaluate_lm(s.model, gate, s.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "completed.json").exists():
        raise RuntimeError("completed mask screen exists")
    identity = {"checkpoint_sha256": sha256_file(args.checkpoint), "gate_sha256": batch_hash(gate),
                "heldout_sha256": batch_hash(ctx["eval_batches"]), "scoring_sha256": batch_hash(scoring),
                "script_sha256": sha256_file(Path(__file__)), "seed": 42}
    configs = [(k, d) for k in (0., 2.) for d in (0., 1., 4.)]
    targets = [.7, .75, .8]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(args.out_dir / "run-manifest.json", {"identity": identity, "configs": configs, "targets": targets,
               "baseline_gate": baseline, "scoring_indices": scoring_indices, "gate_indices": gate_indices,
               "code_commit": commit, "selection": "one mask per score: highest sparsity within 1% gate PPL; no validation/task result used"})
    mean, m2, losses = grad_moments(s, scoring, True)
    total = sum(x.numel() for x in mean)
    scale_score = sum(float(x.abs().double().sum()) for x in mean) / total
    scale_residual = sum(float(s.delta[n].double().square().sum()) for n in s.names) / total
    selected, all_points = [], []
    for penalty, damping in configs:
        tag = f"signed_k{penalty:g}_d{damping:g}"
        scores = {n: mean[i] + penalty * (m2[i] / 63).clamp_min(0).sqrt() / 8
                  + damping * scale_score / max(scale_residual, 1e-30) * s.delta[n].to(s.device).square()
                  for i, n in enumerate(s.names)}
        minimum = min(float(v.min()) for v in scores.values())
        if minimum < 0:
            scores = {n: v - minimum for n, v in scores.items()}
        folder = args.out_dir / tag
        folder.mkdir(exist_ok=True)
        points, best = [], None
        for sparsity, keep, unused in global_keep_masks(scores, s.names, targets, s.device):
            s.apply(keep)
            quality = evaluate_lm(s.model, gate, s.device)
            change = 100 * math.expm1(quality["loss"] - baseline["loss"])
            point = {"sparsity": sparsity, "gate": quality, "gate_ppl_change_percent": change, "passed_gate": change <= 1.}
            points.append(point)
            if point["passed_gate"]:
                path = folder / f"p{sparsity:g}-mask.pt"
                torch.save({n: m.cpu() for n, m in keep.items()}, path)
                best = {"identity_common": {**identity, "sparsity": sparsity}, "mask_path": str(path.resolve()),
                        "mask_sha256": sha256_file(path), "config": [1, 64, penalty], "backward_budget": 64,
                        "selector_definition": {"signed_first_order": True, "penalty": penalty, "quadratic_damping": damping,
                                                "damping_scale": scale_score / max(scale_residual, 1e-30)},
                        "gate": quality, "gate_ppl_change_percent": change, "code_commit": commit}
            print("MASK_GATE", tag, sparsity, change, flush=True)
        if best is not None:
            write_json(folder / "selected.json", best)
            selected.append({"tag": tag, "selected": str((folder / "selected.json").resolve()),
                             "sparsity": best["identity_common"]["sparsity"], "gate_ppl_change_percent": best["gate_ppl_change_percent"]})
        write_json(folder / "points.json", {"identity": identity, "selector": [penalty, damping], "points": points})
        all_points.append({"tag": tag, "points": points})
        del keep, scores
        s.apply(None)
    write_json(args.out_dir / "completed.json", {"status": "completed", "identity": identity, "selected": selected,
               "all_points": all_points, "code_commit": commit})
    print("MASK_SCREEN_COMPLETE", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
