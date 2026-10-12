"""Calibrate signed damage and residual damping on reconstructed V100 controls.

One seed, one checkpoint and one fixed configuration over the whole reported
grid. Selection uses training calibration; validation is measured afterwards.
"""
from pathlib import Path
import argparse
import io
import json
import math
import random
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", choices=("P160_R1", "G2_R1", "G2W"), required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--previous", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from run_residual_sweep_small import load_regime, global_keep_masks
    from run_residual_ablation_small import grad_moments
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, set_seed, sha256_file, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "completed.json").exists():
        print("SKIP completed signed calibration", flush=True)
        return
    set_seed(42)
    ctx = load_regime(args.regime, args.assets / "checkpoints", args.assets, "cuda:0")
    s, pool = ctx["sweep"], ctx["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    offset = len(plan.training_segments[0])
    used = plan.selected_pool_indices[offset:offset + 16]
    gate_indices = [i for i in range(len(pool)) if i not in set(used)][:64]
    extras = [i for i in range(len(pool)) if i not in set(used + gate_indices)]
    random.Random(42).shuffle(extras)
    scoring = plan.scoring_batches[0] + [pool[i] for i in extras[:48]]
    gate = [pool[i] for i in gate_indices]
    prior = json.loads((args.previous / "run-manifest.json").read_text())
    identity = {"seed": 42, "regime": args.regime,
                "checkpoint_sha256": ctx["provenance"]["checkpoint_sha256"],
                "scoring_sha256": batch_hash(scoring), "gate_sha256": batch_hash(gate),
                "heldout_sha256": batch_hash(ctx["eval_batches"]),
                "script_sha256": sha256_file(Path(__file__))}
    for key in ("checkpoint_sha256", "scoring_sha256", "gate_sha256", "heldout_sha256"):
        if identity[key] != prior["identity"][key]:
            raise RuntimeError("previous control input differs: " + key)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    targets = [.3, .5, .6, .65, .7, .75]
    write_json(args.out_dir / "run-manifest.json", {"identity": identity, "code_commit": commit,
               "targets": targets, "baseline_gate": prior["baseline_gate"],
               "baseline_heldout": ctx["uncompressed"], "gpu": torch.cuda.get_device_name(0),
               "selection": "one configuration: largest calibration 1% frontier, then mean calibration loss",
               "scope": "signed/absolute first-order attribution with variability and quadratic residual damping"})
    mean, m2, losses = grad_moments(s, scoring, True)
    total = sum(x.numel() for x in mean)
    scale_score = sum(float(x.abs().double().sum()) for x in mean) / total
    scale_residual = sum(float(s.delta[n].double().square().sum()) for n in s.names) / total
    candidates = []
    def make_scores(config):
        signed, penalty, damping = config
        scores = {}
        for i, n in enumerate(s.names):
            d = s.delta[n].to(s.device)
            if damping == "magnitude":
                value = d.square()
            else:
                value = mean[i] if signed else mean[i].abs()
                value = value + penalty * (m2[i] / 63).clamp_min(0).sqrt() / 8
                value = value + damping * scale_score / max(scale_residual, 1e-30) * d.square()
            scores[n] = value
        minimum = min(float(x.min()) for x in scores.values())
        return {n: x - min(minimum, 0.) for n, x in scores.items()}
    # Magnitude is an explicit calibration candidate; it is never relabelled as a new improvement.
    configs = [(signed, penalty, damping) for signed in (False, True)
               for penalty in (0., 2.) for damping in (0., 1., 4.)]
    configs.append((False, 0., "magnitude"))
    for signed, penalty, damping in configs:
        tag = f"{'signed' if signed else 'absolute'}_k{penalty:g}_d{damping}"
        path = args.out_dir / (tag + ".json")
        if path.exists():
            rec = json.loads(path.read_text())
            if {k: v for k, v in rec["identity"].items() if k != "script_sha256"} != {k: v for k, v in identity.items() if k != "script_sha256"}:
                raise RuntimeError("existing candidate identity differs")
        else:
            scores = make_scores((signed, penalty, damping))
            points = {}
            for p, keep, _ in global_keep_masks(scores, s.names, targets, s.device):
                s.apply(keep)
                value = evaluate_lm(s.model, gate, s.device)
                points[str(p)] = {"gate": value, "gate_ppl_change_percent":
                    100 * math.expm1(value["loss"] - prior["baseline_gate"]["loss"])}
                print(f"CAL {args.regime} {tag} P={p} change={points[str(p)]['gate_ppl_change_percent']:.4f}", flush=True)
            s.apply(None)
            rec = {"status": "completed", "identity": identity, "config": [signed, penalty, damping],
                   "points": points, "backward_budget": 64, "code_commit": commit}
            write_json(path, rec)
            del scores
        candidates.append((tag, rec))
    for path in args.previous.glob("calibration*.json"):
        old = json.loads(path.read_text())
        candidates.append(("previous_" + path.stem, {**old, "points": {p: v for p, v in old["points"].items() if float(p) in targets}}))
    def key(item):
        points = item[1]["points"]
        feasible = [float(p) for p, v in points.items() if v["gate_ppl_change_percent"] <= 1]
        return (max(feasible, default=0), -sum(v["gate"]["loss"] for v in points.values()) / len(points))
    tag, selected = max(candidates, key=key)
    folder = args.previous if tag.startswith("previous_") else args.out_dir
    points = {}
    if tag.startswith("previous_"):
        def selected_masks():
            for p, point in selected["points"].items():
                mask = folder / point["mask"]["file"]
                if sha256_file(mask) != point["mask"]["sha256"]:
                    raise RuntimeError("selected mask hash differs")
                yield p, torch.load(io.BytesIO(zlib.decompress(mask.read_bytes())), map_location="cpu", weights_only=True)
        masks = selected_masks()
    else:
        scores = make_scores(selected["config"])
        masks = ((str(p), keep) for p, keep, _ in global_keep_masks(scores, s.names, targets, s.device))
    for p, keep in masks:
        s.apply(keep)
        value = s.evaluate()
        points[p] = {"evaluation": value, "ppl_change_percent":
            100 * math.expm1(value["val"] - ctx["uncompressed"]["val"])}
    write_json(args.out_dir / "selected.json", {"status": "completed", "identity": identity,
               "config_tag": tag, "config": selected["config"], "points": points, "code_commit": commit,
               "selection_rule": "largest calibration 1% frontier, then mean calibration loss; validation never selects"})
    write_json(args.out_dir / "completed.json", {"status": "completed", "selected": tag})
    print("SELECTED " + tag + " " + json.dumps(points), flush=True)


if __name__ == "__main__":
    main()
