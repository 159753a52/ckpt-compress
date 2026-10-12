"""Predict actual fixed-mask loss changes with shared output-probe prefixes.

Scalar directional curvature is compared with exact GN for each mask. This
diagnostic never selects a mask using heldout quality or task accuracy.
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
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--legacy-mask-record", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from run_residual_sweep_small import REGIMES, load_regime, global_keep_masks
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, set_seed, sha256_file, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from experiments.lib.residual_orthogonal_probes import orthogonal_scores
    from experiments.lib.residual_probe_reference import exact_gn_product
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "completed.json").exists():
        print("SKIP completed mask probe diagnostic", flush=True)
        return
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    s = ctx["sweep"]
    plan = partition_repeated_seed_batches(ctx["pool"], 100, 1, 16, 42)
    scoring = plan.scoring_batches[0][:4]
    tokens = [int((batch["labels"][:, 1:] != -100).sum()) for batch in scoring]
    batch_weights = [count / sum(tokens) for count in tokens]
    selected = json.loads(args.selected.read_text())
    legacy = json.loads(args.legacy_mask_record.read_text())
    masks = {}
    for p, keep, _ in global_keep_masks({n: s.delta[n].abs() for n in s.names}, s.names, [.5, .7], s.device):
        masks[f"ExCP_p{p}"] = {n: m.cpu() for n, m in keep.items()}
    mask = Path(selected["mask_path"])
    if not mask.is_absolute():
        mask = args.selected.parent / mask.name
    if sha256_file(mask) != selected["mask_sha256"]:
        raise RuntimeError("selected mask hash differs")
    masks["DACP_tuned_p0.7"] = torch.load(mask, map_location="cpu", weights_only=True)
    oldmask = args.legacy_mask_record.parent / legacy["mask"]["file"]
    if sha256_file(oldmask) != legacy["mask"]["sha256"]:
        raise RuntimeError("legacy mask hash differs")
    masks["DACP_previous_p0.7"] = torch.load(oldmask, map_location="cpu", weights_only=True)
    baseline = evaluate_lm(s.model, scoring, s.device)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    identity = {"checkpoint_sha256": ctx["provenance"]["checkpoint_sha256"],
                "scoring_sha256": batch_hash(scoring), "heldout_sha256": batch_hash(ctx["eval_batches"]),
                "script_sha256": sha256_file(Path(__file__)), "seed": 42, "scoring_batches": 4,
                "orthogonal_library_sha256": sha256_file(ROOT / "experiments/lib/residual_orthogonal_probes.py")}
    identity["batch_weights"] = batch_weights
    manifest = {"identity": identity, "code_commit": commit, "baseline": baseline,
                "gpu": torch.cuda.get_device_name(0), "masks": {}, "prefixes": [1, 2, 4, 8, 16, 32],
                "definition": "g^T e + 0.5 * mean_j (w_j^T e)^2; e is the actual omitted residual",
                "selection": "fixed masks, no heldout selection"}
    for tag, keep in masks.items():
        started = time.perf_counter()
        e = {n: -s.delta[n] * (~keep[n]) for n in s.names}
        exact = exact_gn_product(s.model, scoring, s.names, e, s.device, batch_weights=batch_weights)
        q = .5 * sum(float((e[n].to(s.device).double() * exact[n].double()).sum()) for n in s.names)
        del exact
        s.apply(keep)
        actual = evaluate_lm(s.model, scoring, s.device)
        validation = s.evaluate()
        s.apply(None)
        manifest["masks"][tag] = {"exact_directional_quadratic": q,
                 "actual_delta_loss": actual["loss"] - baseline["loss"], "actual_scoring": actual,
                 "validation": validation, "exact_gn_seconds": time.perf_counter() - started}
        print(f"EXACT {tag} Q={q:.8g} actual={actual['loss']-baseline['loss']:.8g}", flush=True)
    write_json(args.out_dir / "run-manifest.json", manifest)
    cumulative = {tag: [] for tag in masks}
    linear = {}
    prefixes = []
    def direction(count, values, gradient, meta):
        for tag, keep in masks.items():
            projection, first = 0., 0.
            for n in s.names:
                e = (-s.delta[n] * (~keep[n])).to(s.device)
                projection += float((values[n].double() * e.double()).sum())
                if count == 1:
                    first += float((gradient[n].double() * e.double()).sum())
            cumulative[tag].append(.5 * projection**2)
            if count == 1:
                linear[tag] = first
        if count not in manifest["prefixes"]:
            return
        record = {"rounds": count, "output_backward_passes": count * len(scoring), "masks": {}}
        for tag, samples in cumulative.items():
            q = sum(samples) / len(samples)
            truth = manifest["masks"][tag]
            prediction = linear[tag] + q
            record["masks"][tag] = {"linear": linear[tag], "quadratic_estimate": q,
               "quadratic_relative_error": abs(q - truth["exact_directional_quadratic"]) / max(abs(truth["exact_directional_quadratic"]), 1e-30),
               "predicted_delta_loss": prediction, "actual_delta_loss": truth["actual_delta_loss"],
               "prediction_absolute_error": abs(prediction - truth["actual_delta_loss"]),
               "first_order_absolute_error": abs(linear[tag] - truth["actual_delta_loss"]),
               "exact_gn_prediction_absolute_error": abs(linear[tag] + truth["exact_directional_quadratic"] - truth["actual_delta_loss"])}
        write_json(args.out_dir / f"mask_m{count}_seed42.json", {"identity": identity, "code_commit": commit, **record})
        prefixes.append(record)
        print("PREFIX " + str(count) + " mean_abs_error=" + str(sum(v["prediction_absolute_error"] for v in record["masks"].values())/len(masks)), flush=True)
    set_seed(42)
    score, meta = orthogonal_scores(s.model, scoring, s.names, s.delta, s.device, 32, on_direction=direction, batch_weights=batch_weights)
    del score
    write_json(args.out_dir / "completed.json", {"status": "completed", "identity": identity,
               "code_commit": commit, "probe": meta, "raw_quadratic_samples": cumulative,
               "diagnostic_scope": "scalar mask damage; not an accuracy gain or a true-loss guarantee"})


if __name__ == "__main__":
    main()
