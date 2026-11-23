#!/usr/bin/env python
"""Matched-step summary of study B under the ExCP protocol, per seed and across seeds.

Gap = validation loss of a run minus that of the no-compression run of the same seed
(``BC_s<seed>_nocomp``) at the same step, in millinats.

- after resumption: the evaluations 10/25/50/100/200 steps after every break (resumption) step;
- checkpoints: the loss of the training weights just before each checkpoint is written, from the
  first checkpoint after the first break to the last step of the run;
- final: the checkpoint at the last step;
- restore damage: loss of the reconstruction minus loss before compression at each break.

Usage: study_b_summary.py --runs <runs dir> [--setting BO BC] [--ours dacpt] [--md out.md]
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

METHODS = ("excp", "dacpt", "dacpf", "dacp", "inshrink", "noise")
LABEL = {"excp": "ExCP", "dacpt": "DACP-G_rho", "dacp": "DACP-G", "dacpf": "DACP-G_rho 4x4", "inshrink": "Inshrinkerator", "noise": "noise"}


def load(run: Path):
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines() if x.strip()]
    cfg = json.loads((run / "config.json").read_text())
    ev = {r["step"]: r["loss"] for r in rows if r["kind"] == "eval"}
    ck = {r["step"]: r for r in rows if r["kind"] == "checkpoint"}
    return {"eval": ev, "ckpt": ck, "cfg": cfg, "done": (run / "DONE").exists(),
            "stop": cfg.get("stop_at") or cfg["total_steps"]}


def summarize(runs: Path, setting: str, seed: int, ours: str):
    base_dir = runs / f"BC_s{seed}_nocomp"
    if not (base_dir / "DONE").exists():
        return None
    base = load(base_dir)
    out = {"seed": seed, "methods": {}}
    for m in METHODS:
        d = runs / f"{setting}_s{seed}_{m}"
        if not (d / "DONE").exists():
            continue
        r = load(d)
        stop = min(r["stop"], base["stop"])
        breaks = [s for s in r["cfg"]["break_steps"] if s < stop]
        after = r["cfg"].get("eval_after", "10,25,50,100,200")
        after = [int(a) for a in str(after).split(",")]
        post = [(b + a, 1e3 * (r["eval"][b + a] - base["eval"][b + a]))
                for b in breaks for a in after if b + a <= stop and b + a in r["eval"] and b + a in base["eval"]]
        first = breaks[0] if breaks else 0
        ck = [(s, 1e3 * (r["ckpt"][s]["pre_loss"] - base["ckpt"][s]["pre_loss"]))
              for s in sorted(r["ckpt"]) if first < s <= stop and s in base["ckpt"]]
        dmg = [1e3 * r["ckpt"][b]["immediate_damage"] for b in breaks]
        rates = [r["ckpt"][s].get("deletion_rate", 0.0) for s in sorted(r["ckpt"]) if s <= stop]
        out["methods"][m] = {"post": post, "ckpt": ck, "damage": dmg, "stop": stop,
                             "final_gap": ck[-1][1] if ck else math.nan,
                             "final_ppl": math.exp(r["ckpt"][stop]["pre_loss"]) if stop in r["ckpt"] else math.nan,
                             "base_ppl": math.exp(base["ckpt"][stop]["pre_loss"]) if stop in base["ckpt"] else math.nan,
                             "rate": sum(rates) / len(rates) if rates else math.nan}
    if "excp" in out["methods"] and ours in out["methods"]:
        a, b = out["methods"]["excp"], out["methods"][ours]
        pa, pb = dict(a["post"]), dict(b["post"])
        ca, cb = dict(a["ckpt"]), dict(b["ckpt"])
        common_p = sorted(set(pa) & set(pb))
        common_c = sorted(set(ca) & set(cb))
        out["paired"] = {"post_n": len(common_p), "post_lower": sum(pb[s] < pa[s] for s in common_p),
                         "ckpt_n": len(common_c), "ckpt_lower": sum(cb[s] < ca[s] for s in common_c)}
    return out


def mean(xs):
    xs = [x for _, x in xs] if xs and isinstance(xs[0], tuple) else xs
    return sum(xs) / len(xs) if xs else math.nan


def fmt_ms(vals):
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{st.mean(vals):.1f} ± {st.stdev(vals):.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--setting", nargs="+", default=["BO", "BC"])
    ap.add_argument("--ours", default="dacpt")
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    lines = []
    for setting in args.setting:
        seeds = sorted({int(p.name.split("_s")[1].split("_")[0]) for p in args.runs.glob(f"{setting}_s*_*")
                        if (p / "DONE").exists()})
        res = [r for r in (summarize(args.runs, setting, s, args.ours) for s in seeds) if r and r["methods"]]
        if not res:
            continue
        lines.append(f"## {setting}\n")
        lines.append("| seed | steps | method | mean deletion rate | after resumption: mean gap | lower than ExCP | "
                     "checkpoints: mean gap | lower than ExCP | final gap | final PPL (no comp.) | restore damage at breaks |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in res:
            pr = r.get("paired", {})
            for m, v in r["methods"].items():
                lo_p = f"{pr['post_lower']}/{pr['post_n']}" if m == args.ours and pr else ""
                lo_c = f"{pr['ckpt_lower']}/{pr['ckpt_n']}" if m == args.ours and pr else ""
                lines.append(f"| {r['seed']} | {v['stop']} | {LABEL[m]} | {v['rate']:.3f} | {mean(v['post']):.1f} | {lo_p} | "
                             f"{mean(v['ckpt']):.1f} | {lo_c} | {v['final_gap']:.1f} | "
                             f"{v['final_ppl']:.2f} ({v['base_ppl']:.2f}) | {'/'.join(f'{x:.0f}' for x in v['damage'])} |")
        full = [r for r in res if all(v["stop"] == r["methods"][next(iter(r["methods"]))]["stop"] for v in r["methods"].values())]
        stops = {next(iter(r["methods"].values()))["stop"] for r in full}
        for stop in sorted(stops, reverse=True):
            grp = [r for r in full if next(iter(r["methods"].values()))["stop"] == stop]
            lines.append(f"\nAcross {len(grp)} seed(s) run to step {stop} (mean ± std over seeds):\n")
            for m in METHODS:
                vs = [r["methods"][m] for r in grp if m in r["methods"]]
                if not vs:
                    continue
                lines.append(f"- {LABEL[m]} ({len(vs)} seeds): after resumption {fmt_ms([mean(v['post']) for v in vs])}, "
                             f"checkpoints {fmt_ms([mean(v['ckpt']) for v in vs])}, final {fmt_ms([v['final_gap'] for v in vs])}")
            pl = [r["paired"] for r in grp if "paired" in r]
            if pl:
                lines.append(f"- {LABEL[args.ours]} lower than ExCP: {sum(p['post_lower'] for p in pl)}/{sum(p['post_n'] for p in pl)} "
                             f"points after resumption, {sum(p['ckpt_lower'] for p in pl)}/{sum(p['ckpt_n'] for p in pl)} checkpoints")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.md:
        args.md.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
