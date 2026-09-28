"""Multi-batch scoring on the 32B checkpoint: does averaging first-order scores
over more scoring batches, or penalizing their batch-to-batch uncertainty,
change which residual entries are kept?

Regime: checkpoint step 200 against the base weights (the regime in which
restoration raises the loss).  For B = 1, 2, 4, ..., B_max scoring sequences
(nested prefixes of a fixed held-out pool) we form

    a_bar = -(1/B) sum_b g_b * d,     sigma^2 = (1/B) sum_b (g_b * d)^2 - a_bar^2

and select masks by  |a_bar|  (mean score)  and  |a_bar| + kappa sigma / sqrt(B)
(uncertainty-protected score: coordinates whose attribution is uncertain are
kept).  Every mask is evaluated immediately on the validation and gate sets.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.lib.probe_coverage import cosine, spearman  # noqa: E402
from experiments.scripts.run_probe_coverage_32b import (  # noqa: E402
    Context,
    Scorer,
    load_ckpt,
    now,
    peaks,
    reset_peaks,
    write_json_new,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/configs/probe_coverage_32b.json"))
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--pool-start", type=int, default=1000)
    ap.add_argument("--b-max", type=int, default=64)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--sparsities", default="0.9,0.95,0.98,0.99,0.995")
    ap.add_argument("--reference", choices=["base", "step100"], default="base")
    ap.add_argument("--override", default="{}")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    for key, value in json.loads(args.override).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    cfg["sparsities"] = [float(x) for x in args.sparsities.split(",")]
    work = Path(args.work_dir)
    ctx = Context(cfg, work)
    names = ctx.pipe.eligible_names
    base = ctx.snapshot(names)
    ref100, theta, _, _, _, _ = load_ckpt(ctx, verify_hash=False)
    ref = base if args.reference == "base" else {n: ref100[n] for n in names}
    del ref100
    ctx.load_params(theta)
    pool = ctx.train_chunks[args.pool_start : args.pool_start + args.b_max]
    if args.pool_start + args.b_max > ctx.cfg["chunks"]["gate_start"]:
        raise SystemExit("scoring pool overlaps the gate set")
    out = work / "scoring_rounds"
    out.mkdir(exist_ok=True)
    s = Scorer(ctx, ref, out / "work")
    rec = {"reference": args.reference, "b_max": args.b_max, "kappa": args.kappa, "sparsities": cfg["sparsities"], "pool": [args.pool_start, args.pool_start + args.b_max]}
    rec["residual"] = s.build_residual()
    ev = ctx.evaluate()
    rec["uncompressed"] = {"val": ev["val"]["loss"], "gate": ev["gate"]["loss"]}
    g_sum = [torch.zeros(d.shape, dtype=torch.float32) for d in s.d]  # host
    sq_sum = [torch.zeros(d.shape, dtype=torch.float32, device=d.device) for d in s.d]
    checkpoints = [b for b in (1, 2, 4, 8, 16, 32, 64, 128) if b <= args.b_max]
    rec["by_b"] = {}
    subs = {}
    pipe = ctx.pipe
    reset_peaks()
    t_grad = 0.0
    for b in range(1, args.b_max + 1):
        t0 = now()
        ids = pool[b - 1 : b]
        pipe.set_requires_grad(names)
        pipe.zero_grad()
        logits = pipe.forward(ids)
        loss, n = pipe.nll_sum(logits, ids)
        (loss / n).backward()
        del logits
        with torch.no_grad():
            for i, name in enumerate(names):
                g = pipe.param(name).grad
                a = g.float() * s.d[i].float()
                sq_sum[i].addcmul_(a, a)
                g_sum[i].add_(g.to("cpu").float())
                del a
        pipe.zero_grad()
        pipe.set_requires_grad([])
        t_grad += now() - t0
        if b not in checkpoints:
            continue
        s.g = [(gs / b).to(torch.bfloat16) for gs in g_sum]  # host, used through g_dev
        entry = {"gradient_s_cumulative": round(t_grad, 2), "evals": {}, "select_s": {}}

        def mean_score(i: int) -> torch.Tensor:
            return (s.g_dev(i).float() * s.d[i].float()).abs()

        def ucb_score(i: int, b=b) -> torch.Tensor:
            a = -s.g_dev(i).float() * s.d[i].float()
            var = (sq_sum[i] / b - a * a).clamp_min(0.0)
            return a.abs() + args.kappa * var.sqrt() / math.sqrt(b)

        for tag, fn in (("mean", mean_score), ("ucb", ucb_score)):
            summ = s.select(f"b{b}_{tag}", fn, weibull=False)
            entry["select_s"][tag] = summ["select_s"]
            subs[(b, tag)] = summ["subsample"]
            for mname, packed in summ["masks"].items():
                sp = mname.rsplit("_", 1)[1]
                e = s.immediate_eval(packed)
                entry["evals"].setdefault(tag, {})[sp] = {"val": e["val"]["loss"], "gate": e["gate"]["loss"]}
            ctx.log("rounds_eval", b=b, tag=tag, val={k: round(v["val"] - rec["uncompressed"]["val"], 5) for k, v in entry["evals"][tag].items()})
        rec["by_b"][str(b)] = entry
    ref_sub = subs[(checkpoints[-1], "mean")]
    for (b, tag), sub in subs.items():
        rec["by_b"][str(b)].setdefault("vs_bmax", {})[tag] = {
            "spearman": spearman(sub, ref_sub),
            "rel_l2": float((sub.double() - ref_sub.double()).norm() / ref_sub.double().norm()),
        }
    rec["peak_gib"] = peaks()
    rec["provenance"] = ctx.prov
    rec["gpu"] = ctx.gpu
    write_json_new(out / f"scoring_rounds_{args.reference}.json", rec)


if __name__ == "__main__":
    main()
