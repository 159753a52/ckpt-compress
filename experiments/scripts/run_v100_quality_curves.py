"""Measure 31 sparsity points for frozen, calibration-selected configurations."""
from pathlib import Path
import argparse
import json
import math
import os
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", choices=("P160_R1", "G2_R1", "G2W"), required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--previous", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from run_residual_sweep_small import REGIMES, load_regime, global_keep_masks
    from run_residual_ablation_small import grad_moments, geometric_schedule, run_iterative
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, set_seed, sha256_file, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from experiments.lib.residual_scoring import compute_block_first_order_scores
    os.environ["DACP_DATA_ROOT"] = str(args.assets.resolve())
    os.environ["PYTHIA_MODEL_PATH"] = str(args.assets / "models" / REGIMES[args.regime]["model"])
    ctx = load_regime(args.regime, args.assets / "checkpoints", args.assets, "cuda:0")
    s, pool = ctx["sweep"], ctx["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    first = plan.scoring_batches[0]
    offset = len(plan.training_segments[0])
    used = plan.selected_pool_indices[offset:offset+16]
    gate_indices = [i for i in range(len(pool)) if i not in set(used)][:64]
    extras = [i for i in range(len(pool)) if i not in set(used + gate_indices)]
    random.Random(42).shuffle(extras)
    scoring = first + [pool[i] for i in extras[:48]]
    selected = json.loads(args.selected.read_text())
    targets = [round(.3 + i*.02, 2) for i in range(31)]
    identity = {"seed": 42, "regime": args.regime, "targets": targets,
                "checkpoint_sha256": ctx["provenance"]["checkpoint_sha256"],
                "scoring_sha256": batch_hash(scoring), "gate_sha256": batch_hash([pool[i] for i in gate_indices]),
                "heldout_sha256": batch_hash(ctx["eval_batches"]), "selected_sha256": sha256_file(args.selected),
                "script_sha256": sha256_file(Path(__file__))}
    for key in ("checkpoint_sha256", "scoring_sha256", "gate_sha256", "heldout_sha256"):
        if identity[key] != selected["identity"][key]:
            raise RuntimeError("frozen selection input differs: " + key)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(args.out_dir / "run-manifest.json", {"identity": identity, "code_commit": commit,
               "frozen_selection": {"file": str(args.selected), "sha256": sha256_file(args.selected),
                                    "config": selected["config"], "damping": selected.get("damping", 0)},
               "baseline": ctx["uncompressed"], "gpu": torch.cuda.get_device_name(0),
               "scope": "31 measured sparsities, no new configuration selection or finetuning"})
    for method in ("DACP", "ExCP", "Inshrinkerator"):
        path = args.out_dir / f"{method}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["identity"] != identity:
                raise RuntimeError("existing curve inputs differ")
        else:
            old = selected if method == "DACP" else json.loads((args.previous / f"{method}.json").read_text())
            for key in ("checkpoint_sha256", "heldout_sha256"):
                if old["identity"][key] != identity[key]:
                    raise RuntimeError("reused curve input differs")
            result = {"status": "running", "identity": identity, "method": method, "points": {}, "code_commit": commit}
            for p in targets:
                if str(p) in old["points"]:
                    result["points"][str(p)] = {**old["points"][str(p)], "reused_sha256":
                        sha256_file(args.selected if method == "DACP" else args.previous / f"{method}.json")}
        missing = [p for p in targets if str(p) not in result["points"]]
        s.apply(None)
        set_seed(42)
        scores = None
        if missing and method == "ExCP":
            scores = {n: s.delta[n].abs() for n in s.names}
        elif missing and method == "Inshrinkerator":
            scores, _ = compute_block_first_order_scores(s.model, first[:8], s.layers, s.cur_cpu, s.device,
                       model_family=s.family, block_parameter_names=s.blocks)
        elif missing and args.regime != "G2W":
            signed, penalty, damping = selected["config"]
            mean, m2, _ = grad_moments(s, scoring, True)
            scale = sum(float(x.abs().double().sum()) for x in mean)
            scale /= max(sum(float(s.delta[n].double().square().sum()) for n in s.names), 1e-30)
            scores = {n: (mean[i] if signed else mean[i].abs()) +
                      penalty*(m2[i]/63).clamp_min(0).sqrt()/8 + damping*scale*s.delta[n].to(s.device).square()
                      for i, n in enumerate(s.names)}
            floor = min(min(float(x.min()) for x in scores.values()), 0.)
            scores = {n: x-floor for n, x in scores.items()}
            del mean, m2
        if not missing:
            masks = iter(())
        elif scores is not None:
            masks = ((p, keep) for p, keep, _ in global_keep_masks(scores, s.names, missing, s.device))
        else:
            def staged_masks():
                stages, batches, penalty = selected["config"]
                damping = selected["damping"]
                for p in missing:
                    stage = 0
                    def scorer(current, unused, keep_prev, seed):
                        nonlocal stage
                        bs = scoring[stage*batches:(stage+1)*batches]
                        stage += 1
                        mean, m2, losses = grad_moments(current, bs, True)
                        values = {n: mean[i].abs()+penalty*(m2[i]/(batches-1)).clamp_min(0).sqrt()/math.sqrt(batches)
                                  for i, n in enumerate(current.names)}
                        residual = {n: current.delta[n].to(current.device).square() for n in current.names}
                        active = {n: keep_prev[n].to(current.device) if keep_prev is not None else 1 for n in current.names}
                        scale = sum(float((values[n]*active[n]).double().sum()) for n in current.names)
                        scale /= max(sum(float((residual[n]*active[n]).double().sum()) for n in current.names), 1e-30)
                        return {n: values[n]+damping*scale*residual[n] for n in current.names}, {"score_losses": losses}
                    set_seed(42)
                    _, _, keep = run_iterative(s, scoring, geometric_schedule(p, stages), scorer, 42, eval_at=[], return_keep=True)
                    yield p, keep
            masks = staged_masks()
        for p, keep in masks:
            s.apply(keep)
            value = s.evaluate()
            result["points"][str(p)] = {"evaluation": value,
                "ppl_change_percent": 100*math.expm1(value["val"]-ctx["uncompressed"]["val"])}
            write_json(path, result)
            print(f"CURVE {args.regime} {method} P={p} change={result['points'][str(p)]['ppl_change_percent']:.5f}", flush=True)
            s.apply(None)
        del scores
        if len(result["points"]) != 31:
            raise RuntimeError("sparsity curve is incomplete")
        result["status"] = "completed"
        write_json(path, result)
    write_json(args.out_dir / "completed.json", {"status": "completed", "identity": identity, "code_commit": commit})


if __name__ == "__main__":
    main()
