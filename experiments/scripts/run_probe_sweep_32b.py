"""Immediate-restoration sweep on the 32B checkpoint: two residual regimes,
seven sparsities, and several curvature estimators.

Regimes: reference = step-100 checkpoint (100-step residual) and reference =
base weights (200-step residual).  For each regime and scorer, masks are
selected by exact global k-smallest at every sparsity and the restored model is
evaluated on validation, gate and scoring sets (no recovery training).

Scorers: |d|, |g*theta|, |g*d|, block HVP (global and Weibull), and probe
curvature products: sampled with coverage tracking (m = 1, 4), deterministic
top-k enumeration (m = 1, 2, 4, 8) and head-tail stratified (m = 2, 4; top-(m-1)
head plus one tail draw, unbiased for the Gauss--Newton matrix).  For block-HVP
global masks at four sparsities the mask-level second-order Taylor change is
computed with an extra block HVP of the masked residual.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.lib.probe_coverage import rademacher, unpack_mask, weighted_token_ce  # noqa: E402
from experiments.scripts.run_probe_coverage_32b import (  # noqa: E402
    Context,
    ProbeRun,
    Scorer,
    load_ckpt,
    now,
    peaks,
    reset_peaks,
    write_json_new,
)

SPARSITIES = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
TAYLOR_SPARSITIES = [0.5, 0.9, 0.98, 0.995]


def eval_masks(s: Scorer, summ: dict, out: dict, tag: str) -> None:
    for name, packed in summ["masks"].items():
        s.apply_mask(packed, -1)
        ev = s.ctx.evaluate()
        ev["score"] = s.pipe.evaluate(s.ctx.score_ids, s.ctx.cfg["chunks"]["eval_batch"])
        s.apply_mask(packed, +1)
        sp = name.rsplit("_", 1)[1]
        out.setdefault(tag, {})[sp] = {
            "val": ev["val"]["loss"],
            "gate": ev["gate"]["loss"],
            "score": ev["score"]["loss"],
            "select_s": summ["select_s"],
        }
    s.ctx.log("sweep_eval", tag=tag, val={k: round(v["val"], 5) for k, v in out[tag].items()})


@torch.no_grad()
def masked_taylor(s: Scorer, packed: List[torch.Tensor], hd: List[torch.Tensor]) -> Dict[str, float]:
    """Q(z) = g.delta + 0.5 delta^T H delta (block HVP), A(z) additive attribution."""
    full_d = s.d
    masked = []
    g_delta = 0.0
    attr = 0.0
    for i, n in enumerate(s.names):
        m = unpack_mask(packed[i].to(full_d[i].device), full_d[i].shape)
        md = torch.where(m, full_d[i], torch.zeros_like(full_d[i]))
        masked.append(md)
        g = s.g_dev(i).float()
        g_delta += float(-(g * md.float()).sum().double())
        a = -g * full_d[i].float() + 0.5 * full_d[i].float() * hd[i].to(g.device)
        attr += float(torch.where(m, a, torch.zeros_like(a)).sum().double())
    s.d = masked
    try:
        with torch.enable_grad():
            s.compute_hvp()
        quad = 0.0
        for i in range(len(s.names)):
            quad += float((masked[i].float() * s.hd[i].to(masked[i].device)).sum().double())
    finally:
        s.d = full_d
        s.hd = None
    return {"first_order": g_delta, "second_order": 0.5 * quad, "taylor2": g_delta + 0.5 * quad, "additive_attribution": attr}


class TopkHeadTail:
    """Deterministic top-k probe rounds with optional head-tail evaluations."""

    def __init__(self, s: Scorer, seed: int):
        self.s = s
        self.pipe = s.pipe
        self.gen = torch.Generator(device=self.pipe.devices[-1])
        self.gen.manual_seed(20_000 + seed)
        self.u = [torch.zeros(d.shape, dtype=torch.float32, device=d.device) for d in s.d]
        self.h = 0
        self.head_mass = torch.zeros(s.positions, dtype=torch.float64, device=self.pipe.devices[-1])

    def _pass(self, tail: bool) -> Dict[str, float]:
        s, pipe = self.s, self.pipe
        pipe.set_requires_grad(s.names)
        pipe.zero_grad()
        t0 = now()
        offset = 0
        for _, ids in s._micro_batches():
            logits = pipe.forward(ids)
            flat, _ = pipe.shifted(logits, ids)
            with torch.no_grad():
                probs = torch.softmax(flat.detach().float(), dim=-1)
                n = probs.shape[0]
                xi = rademacher(n, self.gen, probs.device)
                if not tail:
                    tok = probs.topk(self.h + 1, dim=1).indices[:, self.h]
                    p_y = probs.gather(1, tok[:, None]).reshape(-1)
                    w = xi * p_y.double().sqrt() / math.sqrt(s.positions)
                    self.head_mass[offset : offset + n] += p_y.double()
                else:
                    rest = probs.clone()
                    if self.h:
                        rest.scatter_(1, probs.topk(self.h, dim=1).indices, 0.0)
                    r = rest.double().sum(dim=1).clamp_min(0.0)
                    tok = torch.multinomial(rest.clamp_min(0) + 1e-30, 1, generator=self.gen).reshape(-1)
                    w = xi * r.sqrt() / math.sqrt(s.positions)
                del probs
            weighted_token_ce(flat, tok, w).backward()
            del logits, flat
            offset += n
        with torch.no_grad():
            alpha = 0.0
            for i, name in enumerate(s.names):
                alpha += float((pipe.param(name).grad.float() * s.d[i].float()).sum().double())
        return {"alpha": alpha, "seconds": round(now() - t0, 3)}

    def head_round(self) -> Dict[str, float]:
        rec = self._pass(tail=False)
        with torch.no_grad():
            for i, name in enumerate(self.s.names):
                self.u[i].add_(self.pipe.param(name).grad.float(), alpha=rec["alpha"])
        self.pipe.zero_grad()
        self.pipe.set_requires_grad([])
        self.h += 1
        rec["head_mass_mean"] = float(self.head_mass.mean())
        return rec

    def head_score(self, i: int) -> torch.Tensor:
        d = self.s.d[i].float()
        return (-self.s.g_dev(i).float() * d + 0.5 * d * self.u[i]).abs()

    def tail_select(self, prefix: str):
        rec = self._pass(tail=True)
        alpha = rec["alpha"]
        names = self.s.names

        def score(i: int) -> torch.Tensor:
            d = self.s.d[i].float()
            u = self.u[i] + alpha * self.pipe.param(names[i]).grad.float()
            return (-self.s.g_dev(i).float() * d + 0.5 * d * u).abs()

        summ = self.s.select(prefix, score, weibull=False)
        self.pipe.zero_grad()
        self.pipe.set_requires_grad([])
        return rec, summ


def run_regime(ctx: Context, regime: str, ref: Dict[str, torch.Tensor], theta, out_dir: Path) -> dict:
    ctx.load_params(theta)
    s = Scorer(ctx, ref, out_dir)
    res: Dict[str, object] = {"regime": regime, "evals": {}, "costs": {}}
    res["residual"] = s.build_residual()
    ev = ctx.evaluate()
    ev["score"] = ctx.pipe.evaluate(ctx.score_ids, ctx.cfg["chunks"]["eval_batch"])
    res["uncompressed"] = {"val": ev["val"]["loss"], "gate": ev["gate"]["loss"], "score": ev["score"]["loss"]}
    res["gradient"] = s.compute_gradient()
    for name, summ in s.score_baselines().items():
        eval_masks(s, summ, res["evals"], name)
    res["hvp"] = s.compute_hvp()
    summ = s.select("hvp", s.hvp_score, weibull=True)
    res["weibull"] = summ["weibull"]
    glob = {k: v for k, v in summ["masks"].items() if "_global_" in k}
    weib = {k: v for k, v in summ["masks"].items() if "_weibull_" in k}
    eval_masks(s, {"masks": glob, "select_s": summ["select_s"]}, res["evals"], "hvp_global")
    eval_masks(s, {"masks": weib, "select_s": summ["select_s"]}, res["evals"], "dacp")
    hd = s.hd
    res["taylor"] = {}
    for sp in TAYLOR_SPARSITIES:
        t0 = now()
        res["taylor"][str(sp)] = masked_taylor(s, glob[f"hvp_global_{sp}"], hd)
        res["taylor"][str(sp)]["seconds"] = round(now() - t0, 2)
        ctx.log("taylor", regime=regime, sparsity=sp, **res["taylor"][str(sp)])
    s.hd = None
    del hd, glob, weib, summ
    res["gradient_offload_s"] = s.offload_gradient()

    # sampled with coverage tracking
    run = ProbeRun(s, 0, 4, "coverage")
    res["costs"]["sample"] = []
    for m in range(1, 5):
        r = run.round()
        res["costs"]["sample"].append({k: r[k] for k in ("round_s", "mean_mass", "p05_mass", "cumulative_repeat_rate")})
        if m in (1, 4):
            summ = s.select(f"sample_m{m}", run.score, weibull=False)
            eval_masks(s, summ, res["evals"], f"probe_sample_m{m}")
    run.free()

    # top-k and head-tail
    tk = TopkHeadTail(s, 0)
    res["costs"]["topk"] = []
    res["costs"]["tail"] = {}
    for h in range(8):
        if h + 1 in (2, 4):
            rec, summ = tk.tail_select(f"headtail_m{h + 1}")
            res["costs"]["tail"][h + 1] = rec
            eval_masks(s, summ, res["evals"], f"probe_headtail_m{h + 1}")
        rec = tk.head_round()
        res["costs"]["topk"].append(rec)
        if tk.h in (1, 2, 4, 8):
            summ = s.select(f"topk_m{tk.h}", tk.head_score, weibull=False)
            eval_masks(s, summ, res["evals"], f"probe_topk_m{tk.h}")
    tk.u = []
    torch.cuda.empty_cache()
    res["peak_gib"] = peaks()
    s.d = []
    s.g = []
    torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/configs/probe_coverage_32b.json"))
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--regimes", default="step100,base")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cfg["sparsities"] = SPARSITIES
    # A layer cap of 0.99 cannot reach 99.5% global sparsity; the sweep uses
    # 0.999 and still records feasibility under the original 0.95 cap.
    cfg["weibull_cap"] = 0.999
    work = Path(args.work_dir)
    ctx = Context(cfg, work)
    base = ctx.snapshot(ctx.pipe.eligible_names)  # model is still at base weights here
    ref100, theta, _, _, step, _ = load_ckpt(ctx, verify_hash=False)
    out = work / "sweep"
    out.mkdir(exist_ok=True)
    for regime in args.regimes.split(","):
        path = out / f"{regime}.json"
        if path.exists():
            continue
        ref = {n: (ref100[n] if regime == "step100" else base[n]) for n in ctx.pipe.eligible_names}
        t0 = time.time()
        res = run_regime(ctx, regime, ref, theta, out / f"{regime}_work")
        res["seconds"] = round(time.time() - t0, 1)
        res["provenance"] = ctx.prov
        res["gpu"] = ctx.gpu
        res["sparsities"] = SPARSITIES
        write_json_new(path, res)
        ctx.log("sweep_done", regime=regime, seconds=res["seconds"])


if __name__ == "__main__":
    main()
