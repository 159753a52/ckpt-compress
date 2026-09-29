"""Repeated compression-recovery chains on the 32B checkpoint with the
reference fixed to the base (pretrained) weights.

Starting from the saved step-200 checkpoint (parameters and AdamW moments),
cycle c = 1..C

  1. scores the full fine-tuning residual d = theta - base with one method,
  2. restores: pruned coordinates are set to their base values (residual
     sparsity P over the eligible matrices; everything else is lossless),
  3. keeps the optimizer moments unchanged (stored losslessly),
  4. evaluates immediately (validation and gate sets),
  5. trains R steps on the next recovery chunks (200 + (c-1) R, ...) and
     evaluates again.

Unlike the ``chains`` phase of run_probe_coverage_32b.py, the reference is not
advanced to the restored checkpoint: every cycle stores the whole delta to the
base weights as a compressed residual (ExCP's residual-to-pretrained setting).

Methods
  excp_style            |d|, exact global k-smallest
  inshrinkerator_style  |g * theta|, exact global k-smallest
  dacp                  block HVP score with per-layer Weibull allocation
  first_order_ucb       one-shot |mean_b c_b| + kappa std_b(c_b)/sqrt(B),
                        c_b = -g_b * d, B per-sequence gradients
  iterative_ucb         the same score re-linearized over K stages of a
                        geometric schedule P_j = 1 - (1 - P)^(j/K); pruned
                        coordinates are restored before each stage's gradients
  no_compression        training only (reference trajectory)

g and the HVP of the one-shot baselines use the scoring chunks of the config;
the UCB methods use per-sequence gradients on the held-out pool
(train chunks pool_start .. pool_start + B), which must be disjoint from the
recovery chunks and from the gate set.  One JSON per (method, P) is written
once; an existing result is skipped, so a queue can be restarted.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.lib.probe_coverage import packed_popcount, unpack_mask  # noqa: E402
from experiments.scripts.run_probe_coverage_32b import (  # noqa: E402
    Context,
    Scorer,
    build_restored,
    chain_mask,
    load_ckpt,
    now,
    peaks,
    sha256_file,
    reset_peaks,
    run_recovery,
    write_json_new,
)

TINY = 1e-38
METHODS = ("excp_style", "inshrinkerator_style", "dacp", "first_order_ucb", "iterative_ucb", "no_compression")


def geometric_schedule(p: float, k: int) -> List[float]:
    return [round(1.0 - (1.0 - p) ** (j / k), 6) for j in range(1, k + 1)]


def ucb_mask(ctx: Context, ref: Dict[str, torch.Tensor], out: Path, schedule: List[float], pool: torch.Tensor,
             kappa: float, cycle: int) -> Tuple[List[torch.Tensor], dict]:
    """Uncertainty-protected selection (same score and stage loop as
    run_iterative_ucb_32b.py); a one-element schedule is the one-shot scorer."""
    s = Scorer(ctx, ref, out)
    info: Dict[str, object] = {"residual": s.build_residual(), "schedule": schedule, "stages": []}
    pipe = ctx.pipe
    names = s.names
    B = pool.shape[0]
    pruned = None
    for sp in schedule:
        t0 = now()
        reset_peaks()
        if pruned is not None:
            s.apply_mask(pruned, -1)
        g_sum = [torch.zeros(d.shape, dtype=torch.float32) for d in s.d]  # host
        sq_sum = [torch.zeros(d.shape, dtype=torch.float32, device=d.device) for d in s.d]
        loss_sum = 0.0
        t_fb = 0.0
        t_acc = 0.0
        for b in range(B):
            ids = pool[b : b + 1]
            ta = now()
            pipe.set_requires_grad(names)
            pipe.zero_grad()
            logits = pipe.forward(ids)
            loss, n = pipe.nll_sum(logits, ids)
            (loss / n).backward()
            loss_sum += float(loss.detach()) / n
            del logits, loss
            tb = now()
            with torch.no_grad():
                for i, name in enumerate(names):
                    g = pipe.param(name).grad
                    a = g.float() * s.d[i].float()
                    sq_sum[i].addcmul_(a, a)
                    del a
                    g_sum[i].add_(g.to("cpu").float())
            pipe.zero_grad()
            tc = now()
            t_fb += tb - ta
            t_acc += tc - tb
        pipe.set_requires_grad([])
        if pruned is not None:
            s.apply_mask(pruned, +1)
        grad_s = now() - t0
        s.g = [(gs / B).to(torch.bfloat16) for gs in g_sum]
        del g_sum
        prev = pruned

        def score(i: int, prev=prev, sq_sum=sq_sum) -> torch.Tensor:
            a = -s.g_dev(i).float() * s.d[i].float()
            var = (sq_sum[i] / B - a * a).clamp_min(0.0)
            sc = a.abs() + kappa * var.sqrt() / math.sqrt(B) + TINY
            if prev is not None:
                m = unpack_mask(prev[i].to(sc.device), sc.shape)
                sc = torch.where(m, torch.zeros_like(sc), sc)
            return sc

        ctx.cfg["sparsities"] = [sp]
        summ = s.select(f"c{cycle}_ucb_{sp}", score, weibull=False)
        pruned = summ["masks"][f"c{cycle}_ucb_{sp}_global_{sp}"]
        select_s = summ["select_s"]
        if prev is not None:
            kept = sum(packed_popcount(p & q.to(p.device)) for p, q in zip(pruned, prev))
            if kept != sum(packed_popcount(q) for q in prev):
                raise RuntimeError("pruned set is not monotone")
        del sq_sum, summ
        s.g = []
        torch.cuda.empty_cache()
        stage_peak = peaks()
        e = s.immediate_eval(pruned)
        stage = {"sparsity": sp, "val": e["val"]["loss"], "gate": e["gate"]["loss"],
                 "score_loss_at_partial_point": loss_sum / B, "gradient_s": round(grad_s, 2),
                 "fwd_bwd_s": round(t_fb, 2), "accumulate_offload_s": round(t_acc, 2),
                 "select_s": select_s, "stage_s": round(now() - t0, 2), "peak_gib": stage_peak}
        info["stages"].append(stage)
        ctx.log("ucb_stage", cycle=cycle, sparsity=sp, val=round(stage["val"], 5), stage_s=stage["stage_s"],
                gradient_s=stage["gradient_s"])
    info["select_s"] = round(sum(st["select_s"] for st in info["stages"]), 3)
    s.d = []
    torch.cuda.empty_cache()
    return pruned, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "experiments/configs/probe_coverage_32b.json"))
    ap.add_argument("--work-dir")
    ap.add_argument("--out-dir", help="default: <work-dir>/base_chains")
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--sparsity", type=float, default=0.95)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--recovery", type=int, default=50)
    ap.add_argument("--pool-start", type=int, default=1000)
    ap.add_argument("--b", type=int, default=None, help="default 4 (iterative_ucb) / 16 (first_order_ucb)")
    ap.add_argument("--k", type=int, default=4, help="stages of iterative_ucb")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--weibull-cap", type=float, default=0.999)
    ap.add_argument("--hvp-layer-limit", type=int, default=None, help="smoke tests only")
    ap.add_argument("--override", default="{}", help="JSON merged into the config (smoke tests only)")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v

    merge(cfg, json.loads(args.override))
    cfg["weibull_cap"] = args.weibull_cap
    cfg["cycles"] = args.cycles
    cfg["recovery_steps"] = args.recovery
    cfg["sparsities"] = [args.sparsity]
    work = Path(args.work_dir or cfg["work_dir"])
    out = Path(args.out_dir) if args.out_dir else work / "base_chains"
    out.mkdir(parents=True, exist_ok=True)

    m = args.method
    B = args.b if args.b is not None else {"iterative_ucb": 4, "first_order_ucb": 16}.get(m)
    if m == "no_compression":
        cid = "no_compression"
    elif m == "iterative_ucb":
        cid = f"iterative_ucb_B{B}_K{args.k}_{args.sparsity}"
    elif m == "first_order_ucb":
        cid = f"first_order_ucb_B{B}_{args.sparsity}"
    else:
        cid = f"{m}_{args.sparsity}"
    path = out / f"{cid}.json"
    if path.exists():
        print(json.dumps({"event": "skip", "id": cid, "reason": "result exists"}), flush=True)
        return

    torch.manual_seed(0)
    ctx = Context(cfg, work)
    ctx.log("base_chain_start", id=cid, args=vars(args))
    R, C = args.recovery, args.cycles
    ck = cfg["train"]["ckpt_step"]
    rec_end = ck + R * C
    pool = None
    if m in ("iterative_ucb", "first_order_ucb"):
        if args.pool_start < rec_end or args.pool_start + B > cfg["chunks"]["gate_start"]:
            raise SystemExit("scoring pool overlaps the recovery chunks or the gate set")
        pool = ctx.train_chunks[args.pool_start : args.pool_start + B]
    schedule = None
    if m == "iterative_ucb":
        schedule = geometric_schedule(args.sparsity, args.k)
    elif m == "first_order_ucb":
        schedule = [args.sparsity]

    names = ctx.pipe.eligible_names
    t_start = now()
    base_eval = ctx.evaluate()
    base = ctx.snapshot(names)  # the model still holds the base weights
    _, theta0, m1, m2, step0, _ = load_ckpt(ctx, verify_hash=False)
    ctx.load_params(theta0)
    del theta0
    opt = ctx.new_optimizer()
    opt.load_state(m1, m2, step0)
    del m1, m2
    uncompressed = ctx.evaluate()
    ctx.log("eval", tag="uncompressed", val=uncompressed["val"]["loss"], gate=uncompressed["gate"]["loss"])

    stamp = time.strftime("%Y%m%dT%H%M%S")
    partial = out / "partial" / cid
    partial.mkdir(parents=True, exist_ok=True)
    shim = SimpleNamespace(hvp_layer_limit=args.hvp_layer_limit, max_rounds=99)
    cycles = []
    for c in range(1, C + 1):
        tcyc = now()
        crec: Dict[str, object] = {"cycle": c, "step": opt.step_count}
        if m != "no_compression":
            opt.state_to(False)
            torch.cuda.empty_cache()
            reset_peaks()
            tc = now()
            work_dir = out / f"{cid}_work"
            if m in ("iterative_ucb", "first_order_ucb"):
                packed, info = ucb_mask(ctx, base, work_dir, schedule, pool, args.kappa, c)
            else:
                packed, info = chain_mask(ctx, m, base, work_dir, args.sparsity, c, shim)
            crec["score"] = info
            crec["score_total_s"] = round(now() - tc, 2)
            crec["score_peak_gib"] = peaks()
            cur = ctx.snapshot(names)
            tr = now()
            crec["restore"] = build_restored(ctx, base, cur, packed)
            crec["restore_s"] = round(now() - tr, 2)
            del cur, packed
            opt.state_to(True)
        crec["immediate"] = ctx.evaluate()
        reset_peaks()
        t0 = now()
        crec["recovery_chunks"] = [ck + (c - 1) * R, ck + c * R]
        crec["train_log"] = run_recovery(ctx, opt, ck + (c - 1) * R, R)
        crec["recovery_train_s"] = round(now() - t0, 2)
        crec["recovery_peak_gib"] = peaks()
        crec["final"] = ctx.evaluate()
        crec["cycle_s"] = round(now() - tcyc, 2)
        cycles.append(crec)
        write_json_new(partial / f"{stamp}_c{c}.json", crec)
        ctx.log("base_chain_cycle", id=cid, cycle=c, imm_val=crec["immediate"]["val"]["loss"],
                final_val=crec["final"]["val"]["loss"], score_s=crec.get("score_total_s"), cycle_s=crec["cycle_s"])
    result = {
        "id": cid, "method": m, "sparsity": args.sparsity, "reference": "base",
        "recovery_steps": R, "cycles_n": C, "b": B, "k": args.k if m == "iterative_ucb" else None,
        "kappa": args.kappa if pool is not None else None, "schedule": schedule,
        "pool": [args.pool_start, args.pool_start + B] if pool is not None else None,
        "score_chunks": [cfg["chunks"]["score_start"], cfg["chunks"]["score_start"] + cfg["chunks"]["score_count"]],
        "weibull_cap": args.weibull_cap if m == "dacp" else None,
        "start_step": step0, "base_eval": base_eval, "uncompressed": uncompressed,
        "cycles": cycles, "chain_s": round(now() - t_start, 1),
        "data": ctx.data_meta, "model": ctx.model_meta, "provenance": ctx.prov, "gpu": ctx.gpu,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_new(path, result)
    ctx.log("base_chain_done", id=cid, chain_s=result["chain_s"])


if __name__ == "__main__":
    main()
