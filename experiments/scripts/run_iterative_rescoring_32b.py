"""Iterative re-linearized residual pruning on the 32B checkpoint.

The one-shot scores are Taylor attributions at the uncompressed checkpoint, and
the mask-level quadratic model fails beyond ~90% residual sparsity.  Iterative
re-scoring reaches the target sparsity in stages P_1 < P_2 < ... : after stage
j the pruned coordinates are fixed, the gradient is recomputed at the partially
restored point theta_j = b + (1 - z_j) * d, and the next stage prunes the
remaining coordinates with the smallest |g_j * d| (first order) until P_{j+1}.
Each stage costs one ordinary backward over the scoring batch.

Previously pruned coordinates get score 0 and every other score is raised by
1e-38, so they are always selected first and the pruned set grows monotonically.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.lib.probe_coverage import packed_popcount, unpack_mask  # noqa: E402
from experiments.scripts.run_probe_coverage_32b import Context, Scorer, load_ckpt, now, peaks, write_json_new  # noqa: E402

TINY = 1e-38


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/configs/probe_coverage_32b.json"))
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--reference", choices=["base", "step100"], default="base")
    ap.add_argument("--schedule", default="0.5,0.8,0.9,0.95,0.98,0.99,0.995")
    ap.add_argument("--tag", default="")
    ap.add_argument("--override", default="{}")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    for key, value in json.loads(args.override).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    schedule = [float(x) for x in args.schedule.split(",")]
    work = Path(args.work_dir)
    ctx = Context(cfg, work)
    names = ctx.pipe.eligible_names
    base = ctx.snapshot(names)
    ref100, theta, _, _, _, _ = load_ckpt(ctx, verify_hash=False)
    ref = base if args.reference == "base" else {n: ref100[n] for n in names}
    del ref100
    ctx.load_params(theta)
    out = work / "iterative"
    out.mkdir(exist_ok=True)
    s = Scorer(ctx, ref, out / "work")
    rec = {"reference": args.reference, "schedule": schedule, "stages": []}
    rec["residual"] = s.build_residual()
    ev = ctx.evaluate()
    rec["uncompressed"] = {"val": ev["val"]["loss"], "gate": ev["gate"]["loss"]}
    pruned = None  # packed masks of the cumulative pruned set
    t_all = now()
    for sp in schedule:
        t0 = now()
        if pruned is not None:
            s.apply_mask(pruned, -1)
        grad_rec = s.compute_gradient()
        if pruned is not None:
            s.apply_mask(pruned, +1)
        prev = pruned

        def score(i: int, prev=prev) -> torch.Tensor:
            sc = (s.g_dev(i).float() * s.d[i].float()).abs() + TINY
            if prev is not None:
                m = unpack_mask(prev[i].to(sc.device), sc.shape)
                sc = torch.where(m, torch.zeros_like(sc), sc)
            return sc

        ctx.cfg["sparsities"] = [sp]
        summ = s.select(f"iter_{sp}", score, weibull=False)
        pruned = summ["masks"][f"iter_{sp}_global_{sp}"]
        if prev is not None:
            kept_prev = sum(packed_popcount(p & q.to(p.device)) for p, q in zip(pruned, prev))
            total_prev = sum(packed_popcount(q) for q in prev)
            if kept_prev != total_prev:
                raise RuntimeError("pruned set is not monotone")
        e = s.immediate_eval(pruned)
        stage = {
            "sparsity": sp,
            "val": e["val"]["loss"],
            "gate": e["gate"]["loss"],
            "score_loss_at_partial_point": grad_rec["score_loss"],
            "gradient_s": grad_rec["fwd_bwd_s"],
            "select_s": summ["select_s"],
            "stage_s": round(now() - t0, 2),
        }
        rec["stages"].append(stage)
        s.g = []
        torch.cuda.empty_cache()
        ctx.log("iterative_stage", reference=args.reference, sparsity=sp, dval=round((stage["val"] - rec["uncompressed"]["val"]) * 1e3, 2))
    rec["total_s"] = round(now() - t_all, 1)
    rec["peak_gib"] = peaks()
    rec["provenance"] = ctx.prov
    rec["gpu"] = ctx.gpu
    write_json_new(out / f"iterative_{args.reference}{args.tag}.json", rec)


if __name__ == "__main__":
    main()
