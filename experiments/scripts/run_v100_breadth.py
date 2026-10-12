"""Rebuild missing small checkpoints, then calibrate one DACP configuration.

Each model is trained once with seed 42. All three external scoring comparisons
share its checkpoint, eligible coordinates, validation batches and sparsity grid.
"""
from pathlib import Path
import argparse
import json
import io
import math
import os
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", choices=("P160_R1", "G2_R1", "G2W"), required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from run_residual_sweep_small import REGIMES, load_regime, global_keep_masks
    from run_residual_ablation_small import geometric_schedule, grad_moments, run_iterative
    from experiments.lib.residual_runtime import set_seed, sha256_file, batch_hash, evaluate_lm, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches
    model = REGIMES[args.regime]["model"]
    checkpoint = args.assets / "checkpoints" / REGIMES[args.regime]["checkpoint"]
    os.environ["DACP_DATA_ROOT"] = str(args.assets.resolve())
    os.environ["PYTHIA_MODEL_PATH"] = str(args.assets / "models" / model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        import random
        import numpy as np
        from experiments.scripts.finetune import finetune_pythia_410m as ft
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        old_argv = sys.argv
        sys.argv = [str(Path(ft.__file__)), "--total_steps", "1000", "--batch_size", "4", "--seq_length", "512",
                    "--lr", "2e-5", "--save_every", "1000", "--micro_batch_size", "2", "--device", "cuda",
                    "--output_dir", str(checkpoint.parent), "--dataset", REGIMES[args.regime].get("dataset", "alpaca")]
        ft.main()
        sys.argv = old_argv
        torch.cuda.empty_cache()
    else:
        print("REUSE checkpoint " + str(checkpoint), flush=True)
    ctx = load_regime(args.regime, args.assets / "checkpoints", args.assets, "cuda:0")
    s, pool = ctx["sweep"], ctx["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    first = plan.scoring_batches[0]
    offset = len(plan.training_segments[0])
    used = plan.selected_pool_indices[offset:offset + 16]
    gate_indices = [i for i in range(len(pool)) if i not in set(used)][:64]
    extras = [i for i in range(len(pool)) if i not in set(used + gate_indices)]
    import random
    random.Random(42).shuffle(extras)
    scoring = first + [pool[i] for i in extras[:48]]
    gate = [pool[i] for i in gate_indices]
    assert len(scoring) == 64 and not set(used + extras[:48]).intersection(gate_indices)
    baseline_gate = evaluate_lm(s.model, gate, "cuda:0")
    common = {"seed": 42, "regime": args.regime, "checkpoint_sha256": sha256_file(checkpoint),
              "scoring_sha256": batch_hash(scoring), "gate_sha256": batch_hash(gate), "heldout_sha256": batch_hash(ctx["eval_batches"]),
              "script_sha256": sha256_file(Path(__file__))}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(args.out_dir / "run-manifest.json", {"identity": common, "code_commit": commit, "source_provenance": ctx["provenance"],
               "baseline_gate": baseline_gate, "baseline_heldout": ctx["uncompressed"], "gpu": torch.cuda.get_device_name(0),
               "finetune_seed": 42, "finetune_steps": 1000, "platform": "new V100 reconstruction; not the missing H200 checkpoint"})
    targets = [.3, .4, .5, .6, .65, .7, .75, .8, .85, .9, .95, .98, .99, .995]
    configs = [(4, 4, 1.), (4, 16, 2.), (8, 8, 2.)]
    records = []
    for stages, batches, penalty in configs:
        name = f"K{stages}_B{batches}_k{penalty:g}"
        path = args.out_dir / f"calibration_{name}.json"
        identity = {**common, "config": [stages, batches, penalty], "targets": targets}
        if path.exists():
            rec = json.loads(path.read_text())
            if rec["identity"] != identity:
                raise RuntimeError("existing calibration identity differs")
        else:
            points = {}
            for p in targets:
                stage = 0
                def scorer(current, unused, keep_prev, seed):
                    nonlocal stage
                    bs = scoring[stage * batches:(stage + 1) * batches]
                    stage += 1
                    mean, m2, losses = grad_moments(current, bs, True)
                    scores = {n: mean[i].abs() + penalty * (m2[i] / (batches - 1)).clamp_min(0).sqrt() / math.sqrt(batches)
                              for i, n in enumerate(current.names)}
                    return scores, {"score_losses": losses}
                set_seed(42)
                _, detail, keep = run_iterative(s, scoring, geometric_schedule(p, stages), scorer, 42, eval_at=[], return_keep=True)
                s.apply(keep)
                value = evaluate_lm(s.model, gate, "cuda:0")
                mask = args.out_dir / f"{name}_p{p}-mask.pt.zlib"
                stream = io.BytesIO()
                torch.save({n: x.cpu() for n, x in keep.items()}, stream)
                mask.write_bytes(zlib.compress(stream.getvalue(), 6))
                points[str(p)] = {"gate": value, "gate_ppl_change_percent": 100 * math.expm1(value["loss"] - baseline_gate["loss"]),
                                  "mask": {"file": mask.name, "sha256": sha256_file(mask)}, "stage_records": detail}
                del keep
                s.apply(None)
                s.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"CAL {args.regime} {name} P={p} change={points[str(p)]['gate_ppl_change_percent']:.4f}", flush=True)
            rec = {"status": "completed", "identity": identity, "config": [stages, batches, penalty], "points": points, "code_commit": commit}
            write_json(path, rec)
        records.append(rec)
    # Choose a configuration using calibration alone, then measure its full grid.
    def calibration_key(rec):
        feasible = [float(p) for p, v in rec["points"].items() if v["gate_ppl_change_percent"] <= 1.]
        return (max(feasible, default=0.), -rec["points"]["0.7"]["gate"]["loss"])
    selected = max(records, key=calibration_key)
    selected_path = args.out_dir / "DACP.json"
    if not selected_path.exists():
        points = {}
        for p, point in selected["points"].items():
            mask_path = args.out_dir / point["mask"]["file"]
            if sha256_file(mask_path) != point["mask"]["sha256"]:
                raise RuntimeError("selected mask hash differs")
            s.apply(torch.load(io.BytesIO(zlib.decompress(mask_path.read_bytes())), map_location="cpu", weights_only=True))
            value = s.evaluate()
            points[p] = {"evaluation": value, "ppl_change_percent": 100 * math.expm1(value["val"] - ctx["uncompressed"]["val"])}
        write_json(selected_path, {"status": "completed", "identity": common, "config": selected["config"], "points": points,
                   "selection_rule": "largest training-calibration 1% frontier, then lowest calibration loss at 70%; fixed configuration for all reported cells", "code_commit": commit})
    from experiments.lib.residual_scoring import compute_block_first_order_scores
    for method in ("ExCP", "Inshrinkerator"):
        path = args.out_dir / f"{method}.json"
        if path.exists():
            continue
        s.apply(None)
        if method == "ExCP":
            scores = {n: s.delta[n].abs() for n in s.names}
        else:
            scores, _ = compute_block_first_order_scores(s.model, first[:8], s.layers, s.cur_cpu, "cuda:0",
                       model_family=s.family, block_parameter_names=s.blocks)
        points = {}
        for p, keep, _ in global_keep_masks(scores, s.names, targets, "cuda:0"):
            s.apply(keep)
            value = s.evaluate()
            points[str(p)] = {"evaluation": value, "ppl_change_percent": 100 * math.expm1(value["val"] - ctx["uncompressed"]["val"])}
        write_json(path, {"status": "completed", "identity": common, "points": points, "code_commit": commit,
                         "scope": "transferred published scoring rule in shared residual framework"})
        del scores
    write_json(args.out_dir / "completed.json", {"status": "completed", "identity": common, "checkpoint": str(checkpoint), "code_commit": commit})


if __name__ == "__main__":
    main()
