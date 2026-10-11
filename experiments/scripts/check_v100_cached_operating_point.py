"""Measure validation for a cached calibration point and preserve its mask.

Calibration measurements are reused. Mask reconstruction supplies the new
validation measurement and a reusable mask for subsequent recovery training.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    from experiments.lib.residual_runtime import sha256_file, batch_hash, set_seed, write_json
    candidate = json.loads(args.candidate.read_text())
    source_hash = sha256_file(args.candidate)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "validation.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("candidate_sha256") != source_hash or previous.get("status") != "completed":
            raise RuntimeError("existing validation identity conflict")
        print("SKIP " + str(path), flush=True)
        return
    info = candidate["identity"]
    if info["stages"] != 4 or info["batches_per_stage"] != 4 or info["scoring_policy"] != "disjoint4x4":
        raise ValueError("only the cached disjoint K4/B4 point is supported")
    import torch
    from run_residual_sweep_small import REGIMES, load_regime, _jsonable
    from run_residual_ablation_small import geometric_schedule, grad_moments, run_iterative
    from experiments.lib.residual_training import partition_repeated_seed_batches
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    context = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    sweep = context["sweep"]
    scoring = partition_repeated_seed_batches(context["pool"], 100, 1, 16, info["seed"]).scoring_batches[0]
    if info["checkpoint_sha256"] != context["provenance"]["checkpoint_sha256"] or info["scoring_sha256"] != batch_hash(scoring) or info["heldout_sha256"] != batch_hash(context["eval_batches"]):
        raise RuntimeError("cached candidate does not match the input checkpoint/data")
    stage = 0

    def scorer(current, unused, keep_prev, seed):
        nonlocal stage
        selected = scoring[stage * 4:(stage + 1) * 4]
        stage += 1
        mean, m2, losses = grad_moments(current, selected, True)
        scores = {name: mean[i].abs() + (m2[i] / 3).clamp_min(0).sqrt() / 2
                  for i, name in enumerate(current.names)}
        return scores, {"score_losses": losses}

    set_seed(info["seed"])
    _, stages, keep = run_iterative(sweep, scoring, geometric_schedule(info["sparsity"], 4),
                                    scorer, info["seed"], eval_at=[], return_keep=True)
    sweep.apply(keep)
    evaluation = sweep.evaluate()
    mask_path = args.out_dir / "keep-mask.pt"
    torch.save({name: mask.cpu() for name, mask in keep.items()}, mask_path)
    delta = evaluation["val"] - context["uncompressed"]["val"]
    result = {"status": "completed", "purpose": "new validation measurement of cached calibration point",
              "candidate_sha256": source_hash, "candidate_path": str(args.candidate), "identity": info,
              "baseline": context["uncompressed"], "evaluation": evaluation, "delta_loss": delta,
              "relative_ppl_change": math.expm1(delta), "stage_records": stages,
              "calibration_reused": candidate["calibration"],
              "mask": {"file": mask_path.name, "bytes": mask_path.stat().st_size, "sha256": sha256_file(mask_path)},
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    write_json(path, _jsonable(result))
    print(f"RESULT {path} relative_ppl={result['relative_ppl_change']:.6f}", flush=True)


if __name__ == "__main__":
    main()
