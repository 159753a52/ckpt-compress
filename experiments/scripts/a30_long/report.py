#!/usr/bin/env python
"""Summarize the A30 long runs: per-failure damage, recovery and final quality.

Every compressed trajectory is compared with the uncompressed trajectory of the same
study and seed at the same steps.  gap(s) = val loss of the run at step s minus that of
the uncompressed run (nats); at a failure step the run's value is its restored model.
Recovery steps: the first evaluation after a failure at which the gap is within 1%
perplexity (ln 1.01 nats).  Partially finished runs are summarized up to their last
evaluation.  Writes report.md, per_failure.csv and gap_*.png into --out.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

TOL = math.log(1.01)
BASE_STUDY = {"BO": "BC"}  # studies whose no-compression run is another study's
NAME = re.compile(r"^(?P<study>AC|BC|BO|A|B)_s(?P<seed>\d+)_(?P<method>[a-z]+)(?:_p(?P<p>[0-9.]+))?$")
LABEL = {"excp": "ExCP rule", "dacp": "DACP-G", "inshrink": "Inshrinkerator rule", "restart": "Restart from reference",
         "nocomp": "No compression", "noise": "Noise control (0.1% random)",
         "dacpt": "DACP-G (pilot-chosen setting)", "dacpf": "DACP-G (pilot-chosen, 4 x 4 scoring)"}
STUDIES = {"AC": "研究 A（ExCP 协议：链式参考，每 200 步存档、每 1000 步恢复）",
           "BC": "研究 B（ExCP 协议：链式参考，每 250 步存档、每 1250 步恢复）",
           "BO": "研究 B，ExCP 官方默认阈值 α=5e-5（删除数逐存档点由 ExCP 决定，DACP-G 删同样多；基线为研究 B 的不压缩轨迹）",
           "A": "研究 A（固定参考，旧协议）", "B": "研究 B（固定参考，旧协议）"}


def load(run: Path):
    """evals: step -> loss of the training model; fails: records whose restoration changed training
    (failures of the fixed-reference protocol, resume checkpoints of the chain protocol); ckpts: every
    chain checkpoint (reconstruction quality even where training was not resumed)."""
    evals, fails, ckpts = {}, [], []
    mp = run / "metrics.jsonl"
    if not mp.exists():
        return None
    for line in mp.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["kind"] == "eval":
            evals[r["step"]] = r["loss"]
        elif r["kind"] == "checkpoint":
            ckpts.append(r)
            if r.get("break"):
                fails.append(r)
            else:
                evals[r["step"]] = r["pre_loss"]
        else:
            fails.append(r)
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    return {"evals": evals, "fails": fails, "ckpts": ckpts, "summary": summary, "cfg": cfg,
            "done": (run / "DONE").exists()}


def curve(r):
    """step -> loss of the model in training (restored model at failure steps)."""
    c = dict(r["evals"])
    for f in r["fails"]:
        c[f["step"]] = f["post_loss"]
    return c


def ref_curve(r):
    c = dict(r["evals"])
    for f in r["fails"]:
        c[f["step"]] = f["pre_loss"]
    return c


def analyse(run, base):
    rc, bc = curve(run), ref_curve(base)
    common = sorted(set(rc) & set(bc))
    gaps = {s: rc[s] - bc[s] for s in common}
    fsteps = [f["step"] for f in run["fails"]]
    per = []
    for i, f in enumerate(run["fails"]):
        s = f["step"]
        nxt = fsteps[i + 1] if i + 1 < len(fsteps) else None
        after = {s2 - s: g for s2, g in gaps.items() if s <= s2 and (nxt is None or s2 < nxt)}
        rec = next((a for a in sorted(after) if after[a] <= TOL), None)
        before_next = None
        if nxt is not None:
            nf = run["fails"][i + 1]
            if nxt in bc:
                before_next = nf["pre_loss"] - bc[nxt]
        per.append({"step": s, "immediate_damage": f["immediate_damage"],
                    "gap_restored": gaps.get(s), "gap_after": after, "recovery_steps": rec,
                    "gap_before_next_failure": before_next,
                    "kept_residual_l2_fraction": f.get("kept_residual_l2_fraction"),
                    "compress_s": f.get("compress_s")})
    last = max(common) if common else None
    post = [g for s, g in gaps.items() if fsteps and s >= fsteps[0]]
    return {"gaps": gaps, "per": per, "last_step": last, "final_gap": gaps.get(last) if last is not None else None,
            "mean_gap_after_first_failure": st.mean(post) if post else None,
            "mean_abs_gap_after_first_failure": st.mean(abs(g) for g in post) if post else None}


def fmt(x, scale=1e3, nd=1):
    return "—" if x is None else f"{x * scale:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for d in sorted(args.runs.iterdir()):
        m = NAME.match(d.name)
        if m and (d / "metrics.jsonl").exists():
            r = load(d)
            if r:
                runs[(m["study"], int(m["seed"]), m["method"], float(m["p"]) if m["p"] else None)] = r
    lines = ["# A30 长程实验汇总", "",
             "损失差距 = 该轨迹验证损失 − 同种子不压缩轨迹在同一步的验证损失，单位毫奈特（10 毫奈特约等于 1% PPL）。",
             "恢复步数 = 故障后第一次差距回到 1% PPL 以内的评测点（故障后 0/10/25/50/100/200 步及常规评测点）。", ""]
    rows_csv = []
    for study in ("AC", "BC", "BO", "A", "B"):
        keys = sorted(k for k in runs if k[0] == study and k[2] != "nocomp")
        if not keys:
            continue
        lines += [f"## {STUDIES[study]}", ""]
        if study == "B":
            dec = args.runs / "B_pilot_decide" / "decision.json"
            if dec.exists():
                dj = json.loads(dec.read_text())
                lines += [f"彩排判定：删除率 {dj['sparsity']}，κ={dj['kappa']:g}（{dj['reason']}）。", ""]
                lines += ["| P | κ* | 条件a | 条件b | 优势 | t375 ExCP/DACP/重来 | t1125 ExCP/DACP/重来 |", "|---|---|---|---|---|---|---|"]
                for r in dj["rows"]:
                    g = r["gaps"]
                    cells = [f"{fmt(g[t]['excp'])}/{fmt(g[t]['dacp'])}/{fmt(g[t]['restart'])}" for t in sorted(g, key=int)]
                    lines.append(f"| {r['sparsity']} | {r['kappa']:g} | {r['cond_a']} | {r['cond_b']} | {r['adv']:.2f} | " + " | ".join(cells) + " |")
                lines.append("")
            # run-to-run noise without compression: same initialization and data order, nondeterministic kernels
            pre = load(args.runs / "B_pilot_pre") if (args.runs / "B_pilot_pre").exists() else None
            main_nc = runs.get(("B", 42, "nocomp", None))
            noise = []
            if pre and main_nc:
                noise += [(s, main_nc["evals"][s] - pre["evals"][s]) for s in sorted(set(pre["evals"]) & set(main_nc["evals"])) if s > 0]
            c375 = load(args.runs / "B_pilot" / "t375_nocomp") if (args.runs / "B_pilot" / "t375_nocomp").exists() else None
            if pre and c375 and 500 in pre["evals"] and 500 in c375["evals"]:
                noise.append(("500（从第 375 步状态续跑）", c375["evals"][500] - pre["evals"][500]))
            if noise:
                lines += ["随机波动估计（两条独立运行、都不压缩的轨迹在同一步的验证损失差，毫奈特）："
                          + "，".join(f"第 {s} 步 {1e3 * v:+.2f}" for s, v in noise), ""]
        lines += ["| 种子 | 方法 | P | 进度 | 平均立即损伤 | 平均恢复步数 | 下次故障前差距(均值) | 故障后平均差距 | 故障后平均偏离(绝对值) | 最终差距 | 最终 PPL | 全部存档点重建损伤 | 平均删除率 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        agg = defaultdict(list)
        for k in keys:
            base = runs.get((BASE_STUDY.get(study, study), k[1], "nocomp", None))
            if base is None:
                continue
            r = runs[k]
            a = analyse(r, base)
            imm = [p["immediate_damage"] for p in a["per"]]
            recs = [p["recovery_steps"] for p in a["per"] if p["recovery_steps"] is not None]
            unrec = sum(1 for p in a["per"] if p["recovery_steps"] is None)
            bnext = [p["gap_before_next_failure"] for p in a["per"] if p["gap_before_next_failure"] is not None]
            ppl = r["summary"]["final_ppl"] if r["summary"] else None
            base_ppl = base["summary"]["final_ppl"] if base["summary"] else None
            prog = "完成" if r["done"] else f"至第 {a['last_step']} 步"
            rec_txt = (f"{st.mean(recs):.0f}" if recs else "—") + (f"（{unrec} 次未在窗口内恢复）" if unrec else "")
            ppl_txt = f"{ppl:.3f}（基线 {base_ppl:.3f}）" if ppl and base_ppl else "—"
            ck = [c for c in r.get("ckpts", []) if "deletion_rate" in c]
            ck_dmg = fmt(st.mean(c["immediate_damage"] for c in ck)) if ck else "—"
            ck_rate = f"{st.mean(c['deletion_rate'] for c in ck):.3f}" if ck else "—"
            lines.append(f"| {k[1]} | {LABEL.get(k[2], k[2])} | {k[3] if k[3] else '—'} | {prog} | "
                         f"{fmt(st.mean(imm) if imm else None)} | {rec_txt} | {fmt(st.mean(bnext) if bnext else None)} | "
                         f"{fmt(a['mean_gap_after_first_failure'])} | {fmt(a['mean_abs_gap_after_first_failure'])} | {fmt(a['final_gap'])} | {ppl_txt} | {ck_dmg} | {ck_rate} |")
            if r["done"] and base["done"]:
                agg[(k[2], k[3])].append(a["final_gap"])
            for p in a["per"]:
                rows_csv.append({"study": study, "seed": k[1], "method": k[2], "sparsity": k[3], "failure_step": p["step"],
                                 "immediate_damage": p["immediate_damage"], "gap_restored": p["gap_restored"],
                                 **{f"gap_plus{x}": p["gap_after"].get(x) for x in (10, 25, 50, 100, 200)},
                                 "gap_before_next_failure": p["gap_before_next_failure"],
                                 "recovery_steps": p["recovery_steps"], "compress_s": p["compress_s"]})
            plot(args.out, study, k, a)
        if agg:
            lines += ["", "最终差距（已完成的种子，均值±标准差，毫奈特）：", ""]
            for (m, p), v in sorted(agg.items(), key=lambda x: (str(x[0][1]), x[0][0])):
                sd = f"±{1e3 * st.stdev(v):.1f}" if len(v) > 1 else ""
                lines.append(f"- {LABEL.get(m, m)} P={p}：{1e3 * st.mean(v):.1f}{sd}（{len(v)} 个种子）")
        lines.append("")
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if rows_csv:
        with (args.out / "per_failure.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
            w.writeheader()
            w.writerows(rows_csv)
    print((args.out / "report.md").read_text(encoding="utf-8"))


def plot(out, study, key, a):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    colors = {"excp": "#E69F00", "dacp": "#0072B2", "dacpt": "#56B4E9", "inshrink": "#009E73", "restart": "#999999", "noise": "#CC79A7"}
    fig_path = out / f"gap_{study}_s{key[1]}_p{key[3]}.png"
    cache = out / f".plotdata_{study}_s{key[1]}_p{key[3]}.json"
    data = json.loads(cache.read_text()) if cache.exists() else {}
    data[key[2]] = sorted(a["gaps"].items())
    cache.write_text(json.dumps(data))
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=150)
    for m, pts in data.items():
        xs, ys = zip(*pts) if pts else ([], [])
        ax.plot(xs, [1e3 * y for y in ys], lw=0.9, color=colors.get(m, "k"), label=LABEL.get(m, m))
    ax.axhline(1e3 * TOL, lw=0.6, ls="--", color="0.4")
    ax.axhline(0, lw=0.5, color="0.7")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\Delta\mathcal{L}$ vs. no compression (mnats)")
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
