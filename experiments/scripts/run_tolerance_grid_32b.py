"""Fine sparsity grid for one-shot scorers on the 32B checkpoint.

Locates, for each scorer, the highest residual sparsity at which the immediate
validation-loss increase stays within a tolerance (reported offline).  Regime:
step-200 checkpoint against the base weights.  Scorers: |d| (ExCP-style),
|g*theta| (Inshrinkerator-style), |g*d|, block HVP (global) and DACP.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.scripts.run_probe_coverage_32b import Context, Scorer, load_ckpt, peaks, write_json_new  # noqa: E402
from experiments.scripts.run_probe_sweep_32b import eval_masks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/configs/probe_coverage_32b.json"))
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--sparsities", default="0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97")
    ap.add_argument("--override", default="{}")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    for key, value in json.loads(args.override).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    cfg["sparsities"] = [float(x) for x in args.sparsities.split(",")]
    cfg["weibull_cap"] = 0.999
    work = Path(args.work_dir)
    ctx = Context(cfg, work)
    names = ctx.pipe.eligible_names
    base = ctx.snapshot(names)
    _, theta, _, _, _, _ = load_ckpt(ctx, verify_hash=False)
    ctx.load_params(theta)
    out = work / "tolerance_grid"
    out.mkdir(exist_ok=True)
    s = Scorer(ctx, base, out / "work")
    res = {"regime": "base", "sparsities": cfg["sparsities"], "evals": {}}
    res["residual"] = s.build_residual()
    ev = ctx.evaluate()
    ev["score"] = ctx.pipe.evaluate(ctx.score_ids, cfg["chunks"]["eval_batch"])
    res["uncompressed"] = {"val": ev["val"]["loss"], "gate": ev["gate"]["loss"], "score": ev["score"]["loss"]}
    res["gradient"] = s.compute_gradient()
    for name, summ in s.score_baselines().items():
        eval_masks(s, summ, res["evals"], name)
    res["hvp"] = s.compute_hvp()
    summ = s.select("hvp", s.hvp_score, weibull=True)
    glob = {k: v for k, v in summ["masks"].items() if "_global_" in k}
    weib = {k: v for k, v in summ["masks"].items() if "_weibull_" in k}
    eval_masks(s, {"masks": glob, "select_s": summ["select_s"]}, res["evals"], "hvp_global")
    eval_masks(s, {"masks": weib, "select_s": summ["select_s"]}, res["evals"], "dacp")
    res["peak_gib"] = peaks()
    res["provenance"] = ctx.prov
    res["gpu"] = ctx.gpu
    write_json_new(out / "grid_base.json", res)


if __name__ == "__main__":
    main()
