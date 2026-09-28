"""Iterative uncertainty-protected residual pruning on the 32B checkpoint.

Stage j restores the coordinates pruned so far, computes per-sequence
true-label gradients g_b (b = 1..B) at that partially restored point and scores
each surviving coordinate by

    |mean_b c_b| + kappa * std_b(c_b) / sqrt(B),   c_b = -g_b * d,

(``--score ucb``) or by |mean_b c_b| (``--score mean``).  Previously pruned
coordinates keep score 0, so the pruned set grows monotonically.  The stage
schedule is the union of a geometric grid in the kept fraction,
P_j = 1 - (1 - P_max)^(j/K), and the reporting sparsities; every stage is
evaluated immediately on the validation and gate sets.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.lib.probe_coverage import packed_popcount, unpack_mask  # noqa: E402
from experiments.scripts.run_probe_coverage_32b import Context, Scorer, load_ckpt, now, peaks, reset_peaks, write_json_new  # noqa: E402

TINY = 1e-38


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/configs/probe_coverage_32b.json"))
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--reference", choices=["base", "step100"], default="base")
    ap.add_argument("--score", choices=["ucb", "mean"], default="ucb")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--pool-start", type=int, default=1000)
    ap.add_argument("--b", type=int, default=16)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--p-max", type=float, default=0.995)
    ap.add_argument("--report", default="0.5,0.8,0.9,0.95,0.98,0.99,0.995")
    ap.add_argument("--tag", default="")
    ap.add_argument("--override", default="{}")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    for key, value in json.loads(args.override).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    report = [float(x) for x in args.report.split(",")]
    grid = [round(1.0 - (1.0 - args.p_max) ** (j / args.k), 6) for j in range(1, args.k + 1)]
    schedule = sorted(set(grid) | set(report))
    work = Path(args.work_dir)
    ctx = Context(cfg, work)
    names = ctx.pipe.eligible_names
    base = ctx.snapshot(names)
    ref100, theta, _, _, _, _ = load_ckpt(ctx, verify_hash=False)
    ref = base if args.reference == "base" else {n: ref100[n] for n in names}
    del ref100, base
    ctx.load_params(theta)
    if args.pool_start + args.b > ctx.cfg["chunks"]["gate_start"]:
        raise SystemExit("scoring pool overlaps the gate set")
    pool = ctx.train_chunks[args.pool_start : args.pool_start + args.b]
    out = work / "iterative_ucb"
    out.mkdir(exist_ok=True)
    s = Scorer(ctx, ref, out / "work")
    rec = {"reference": args.reference, "score": args.score, "kappa": args.kappa, "b": args.b, "k": args.k,
           "schedule": schedule, "report": report, "pool": [args.pool_start, args.pool_start + args.b], "stages": []}
    rec["residual"] = s.build_residual()
    ev = ctx.evaluate()
    rec["uncompressed"] = {"val": ev["val"]["loss"], "gate": ev["gate"]["loss"]}
    pipe = ctx.pipe
    pruned = None
    reset_peaks()
    t_all = now()
    for sp in schedule:
        t0 = now()
        if pruned is not None:
            s.apply_mask(pruned, -1)
        g_sum = [torch.zeros(d.shape, dtype=torch.float32) for d in s.d]
        sq_sum = [torch.zeros(d.shape, dtype=torch.float32, device=d.device) for d in s.d] if args.score == "ucb" else None
        loss_sum = 0.0
        for b in range(args.b):
            ids = pool[b : b + 1]
            pipe.set_requires_grad(names)
            pipe.zero_grad()
            logits = pipe.forward(ids)
            loss, n = pipe.nll_sum(logits, ids)
            (loss / n).backward()
            loss_sum += float(loss.detach()) / n
            del logits, loss
            with torch.no_grad():
                for i, name in enumerate(names):
                    g = pipe.param(name).grad
                    if sq_sum is not None:
                        a = g.float() * s.d[i].float()
                        sq_sum[i].addcmul_(a, a)
                        del a
                    g_sum[i].add_(g.to("cpu").float())
            pipe.zero_grad()
        pipe.set_requires_grad([])
        if pruned is not None:
            s.apply_mask(pruned, +1)
        grad_s = now() - t0
        s.g = [(gs / args.b).to(torch.bfloat16) for gs in g_sum]
        del g_sum
        prev = pruned
        B = args.b

        def score(i: int, prev=prev) -> torch.Tensor:
            a = -s.g_dev(i).float() * s.d[i].float()
            sc = a.abs()
            if sq_sum is not None:
                var = (sq_sum[i] / B - a * a).clamp_min(0.0)
                sc = sc + args.kappa * var.sqrt() / math.sqrt(B)
            sc = sc + TINY
            if prev is not None:
                m = unpack_mask(prev[i].to(sc.device), sc.shape)
                sc = torch.where(m, torch.zeros_like(sc), sc)
            return sc

        ctx.cfg["sparsities"] = [sp]
        summ = s.select(f"iu_{sp}", score, weibull=False)
        pruned = summ["masks"][f"iu_{sp}_global_{sp}"]
        if prev is not None:
            kept = sum(packed_popcount(p & q.to(p.device)) for p, q in zip(pruned, prev))
            if kept != sum(packed_popcount(q) for q in prev):
                raise RuntimeError("pruned set is not monotone")
        del sq_sum
        s.g = []
        torch.cuda.empty_cache()
        e = s.immediate_eval(pruned)
        stage = {"sparsity": sp, "val": e["val"]["loss"], "gate": e["gate"]["loss"],
                 "score_loss_at_partial_point": loss_sum / args.b, "gradient_s": round(grad_s, 2),
                 "select_s": summ["select_s"], "stage_s": round(now() - t0, 2)}
        rec["stages"].append(stage)
        ctx.log("iu_stage", reference=args.reference, score=args.score, sparsity=sp,
                dval=round((stage["val"] - rec["uncompressed"]["val"]) * 1e3, 2), stage_s=stage["stage_s"])
    rec["total_s"] = round(now() - t_all, 1)
    rec["peak_gib"] = peaks()
    rec["provenance"] = ctx.prov
    rec["gpu"] = ctx.gpu
    write_json_new(out / f"iu_{args.reference}_{args.score}_B{args.b}_K{args.k}{args.tag}.json", rec)


if __name__ == "__main__":
    main()
