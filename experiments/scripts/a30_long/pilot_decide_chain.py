#!/usr/bin/env python
"""Choose the DACP-G configuration of the ExCP-protocol study B from its resume pilot.

Every pilot run starts from the same no-compression state, compresses the residual of the next
checkpoint interval, resumes from the reconstruction at the break step and trains on (deterministic
kernels, so all runs agree bit for bit before the break).  The rule, fixed before the pilot ran: the
DACP-G variant with the smallest mean validation-loss gap to the uncompressed continuation over the
evaluations after the break (break + 10, 25, 50, 100, 200 steps) is chosen.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def evals(run: Path):
    out, rec = {}, {}
    for line in (run / "metrics.jsonl").read_text().splitlines():
        x = json.loads(line)
        if x["kind"] == "eval":
            out[x["step"]] = x["loss"]
        elif x["kind"] == "checkpoint":
            out[x["step"]] = x["post_loss"] if x.get("break") else x["pre_loss"]
            rec[x["step"]] = x
    return out, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--break-step", type=int, default=3750)
    ap.add_argument("--after", default="10,25,50,100,200")
    ap.add_argument("--sparsity", type=float, default=0.88)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    after = [args.break_step + int(a) for a in args.after.split(",")]
    base, _ = evals(args.pilot / "nocomp")
    rows = []
    for run in sorted(p for p in args.pilot.iterdir() if (p / "DONE").exists() and p.name != "nocomp"):
        ev, rec = evals(run)
        cfg = json.loads((run / "config.json").read_text())
        gaps = {s: ev[s] - base[s] for s in after if s in ev and s in base}
        if len(gaps) != len(after):
            continue
        b = rec.get(args.break_step, {})
        rows.append({"name": run.name, "method": cfg["method"], "kappa": cfg.get("kappa"),
                     "protect_top": cfg.get("protect_top", 0.0), "rho": cfg.get("rho", 0.0),
                     "mean_gap": st.mean(gaps.values()),
                     "gaps": gaps, "immediate_damage": b.get("immediate_damage"),
                     "kept_residual_l2_fraction": b.get("kept_residual_l2_fraction")})
    dacp = [r for r in rows if r["method"] == "dacp"]
    if not dacp:
        raise SystemExit("no finished DACP-G pilot")
    best = min(dacp, key=lambda r: r["mean_gap"])
    excp = next((r for r in rows if r["name"] == "excp"), None)
    reason = (f"{best['name']} has the smallest mean gap after the break among the DACP-G variants "
              f"({1e3 * best['mean_gap']:.1f} mnats"
              + (f"; ExCP rule {1e3 * excp['mean_gap']:.1f}" if excp else "") + ")")
    dec = {"sparsity": args.sparsity, "kappa": best["kappa"], "protect_top": best["protect_top"], "rho": best["rho"],
           "variant": best["name"], "reason": reason, "rows": rows}
    args.out.write_text(json.dumps(dec, indent=1))
    for r in sorted(rows, key=lambda r: r["mean_gap"]):
        print(f"{r['name']:>16s}  mean gap {1e3 * r['mean_gap']:7.1f}  damage "
              f"{1e3 * (r['immediate_damage'] or 0):7.1f}  kept L2 {r['kept_residual_l2_fraction']}")
    print(reason)


if __name__ == "__main__":
    main()
