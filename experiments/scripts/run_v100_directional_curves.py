"""Measure dense loss-response curves and separate true-Hessian from GN bias.

Twenty-one true measurements per fixed mask, one seed and one scoring set.
No interpolation substitutes for measurements; predictors use derivatives at
the original checkpoint and are evaluated along the same deletion direction.
"""
from pathlib import Path
import argparse
import json
import math
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--legacy-mask-record", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from run_residual_sweep_small import REGIMES, load_regime, global_keep_masks
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from experiments.lib.residual_runtime import evaluate_lm, lm_loss, batch_hash, sha256_file, write_json
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "completed.json").exists():
        print("SKIP completed directional curves", flush=True)
        return
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    s = ctx["sweep"]
    s.model.eval()
    scoring = partition_repeated_seed_batches(ctx["pool"], 100, 1, 16, 42).scoring_batches[0][:4]
    reference = json.loads(args.reference.read_text())
    if reference["identity"]["checkpoint_sha256"] != ctx["provenance"]["checkpoint_sha256"] or reference["identity"]["scoring_sha256"] != batch_hash(scoring):
        raise RuntimeError("directional reference inputs differ")
    masks = {}
    for p, keep, _ in global_keep_masks({n: s.delta[n].abs() for n in s.names}, s.names, [.5, .7], s.device):
        masks[f"ExCP_p{p}"] = {n: m.cpu() for n, m in keep.items()}
    chosen = json.loads(args.selected.read_text())
    mask = Path(chosen["mask_path"])
    if not mask.is_absolute():
        mask = args.selected.parent / mask.name
    if sha256_file(mask) != chosen["mask_sha256"]:
        raise RuntimeError("tuned mask differs")
    masks["DACP_tuned_p0.7"] = torch.load(mask, map_location="cpu", weights_only=True)
    legacy = json.loads(args.legacy_mask_record.read_text())
    mask = args.legacy_mask_record.parent / legacy["mask"]["file"]
    if sha256_file(mask) != legacy["mask"]["sha256"]:
        raise RuntimeError("prior mask differs")
    masks["DACP_previous_p0.7"] = torch.load(mask, map_location="cpu", weights_only=True)
    baseline = evaluate_lm(s.model, scoring, s.device)
    if abs(baseline["loss"] - reference["baseline"]["loss"]) > 1e-6:
        raise RuntimeError(f"original scoring loss differs: {baseline['loss']} vs {reference['baseline']['loss']}")
    combined = {key: torch.cat([b[key] for b in scoring]) for key in scoring[0]}
    variables = [dict(s.model.named_parameters())[n] for n in s.names]
    directions = {tag: {n: (-s.delta[n]*(~keep[n])).to(s.device) for n in s.names} for tag, keep in masks.items()}
    derivatives = {}
    with sdpa_kernel(SDPBackend.MATH):
        loss = lm_loss(s.model, combined, s.device)
        g = torch.autograd.grad(loss, variables, create_graph=True)
        for index, (tag, e) in enumerate(directions.items()):
            contraction = sum((value*e[n]).sum() for n, value in zip(s.names, g))
            h = torch.autograd.grad(contraction, variables, retain_graph=index < len(directions)-1)
            derivatives[tag] = {"linear": sum(float((value.detach().double()*e[n].double()).sum()) for n, value in zip(s.names, g)),
                "true_hessian_quadratic": .5*sum(float((value.double()*e[n].double()).sum()) for n, value in zip(s.names, h)),
                "gn_quadratic": reference["masks"][tag]["exact_directional_quadratic"]}
            del contraction, h
    del g, loss
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    identity = {"checkpoint_sha256": ctx["provenance"]["checkpoint_sha256"], "scoring_sha256": batch_hash(scoring),
                "reference_sha256": sha256_file(args.reference), "script_sha256": sha256_file(Path(__file__)),
                "seed": 42, "scales": [i/20 for i in range(21)]}
    for tag, e in directions.items():
        points = []
        for t in identity["scales"]:
            with torch.no_grad():
                for n in s.names:
                    interpolated = torch.lerp(s.cur_dev[n], s.ref_dev[n], t)
                    s.params[n].copy_(torch.where(masks[tag][n].to(s.device), s.cur_dev[n], interpolated))
            actual = evaluate_lm(s.model, scoring, s.device)
            d = derivatives[tag]
            delta = actual["loss"] - baseline["loss"]
            points.append({"scale": t, "actual_delta_loss": delta, "actual": actual,
                "first_order": t*d["linear"],
                "true_hessian_prediction": t*d["linear"] + t*t*d["true_hessian_quadratic"],
                "gn_prediction": t*d["linear"] + t*t*d["gn_quadratic"]})
        s.apply(masks[tag])
        direct = evaluate_lm(s.model, scoring, s.device)
        direct_delta = direct["loss"] - baseline["loss"]
        endpoint = {"interpolated_delta_loss": points[-1]["actual_delta_loss"], "direct_mask_delta_loss": direct_delta,
                    "reference_delta_loss": reference["masks"][tag]["actual_delta_loss"]}
        write_json(args.out_dir / (tag + "_endpoint.json"), endpoint)
        if abs(direct_delta - endpoint["reference_delta_loss"]) > 1e-6 or abs(endpoint["interpolated_delta_loss"]-direct_delta) > 1e-6:
            raise RuntimeError(f"measured endpoint differs for {tag}: {endpoint}")
        write_json(args.out_dir / (tag + ".json"), {"status": "completed", "identity": identity, "tag": tag,
                   "derivatives": derivatives[tag], "points": points, "code_commit": commit})
        print("CURVE " + tag + " Hessian_error=" + str(abs(points[-1]["true_hessian_prediction"]-points[-1]["actual_delta_loss"])), flush=True)
    s.apply(None)
    write_json(args.out_dir / "completed.json", {"status": "completed", "identity": identity,
               "actual_measurements": len(masks)*21, "code_commit": commit,
               "scope": "loss-response diagnosis; fractional residual reversion is not a compressed checkpoint"})


if __name__ == "__main__":
    main()
