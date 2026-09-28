"""Summarize run_paper_experiments.py job records of the Pythia-410M campaign.

Every number printed is read from the job JSON files written by the runner.
For each (workload, prune ratio, K) table and method:

* imm1 / immK: perplexity right after compression in cycle 1 / cycle K,
* final: perplexity after the final training segment,
* mean +- sample std over seeds,
* d_nc: mean paired difference (method - no_compression) of the final PPL,
* vs excp / vs insh: paired sign counts of the final PPL over shared seeds
  written as wins/ties/losses (win = lower perplexity than the baseline).

Usage: python summarize_probe_pythia410m_campaign.py <suite_dir> [...] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

BASELINES = ("excp_style", "inshrinkerator_style")


def _mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.3f} +- {std:.3f}"


def _sign(method: dict[int, float], baseline: dict[int, float]) -> str:
    seeds = sorted(set(method) & set(baseline))
    if not seeds:
        return "n/a"
    wins = sum(method[s] < baseline[s] for s in seeds)
    ties = sum(method[s] == baseline[s] for s in seeds)
    return f"{wins}/{ties}/{len(seeds) - wins - ties}"


def load_records(suite_dirs: list[Path]) -> dict[tuple, list[dict]]:
    tables: dict[tuple, list[dict]] = defaultdict(list)
    for suite in suite_dirs:
        for path in sorted(suite.glob("*_k*.json")):
            job = json.loads(path.read_text(encoding="utf-8"))
            config = job["config"]
            key = (
                suite.name,
                config["name"],
                config["checkpoint_step"],
                config["learning_rate"],
                config["prune_ratio"],
                config["recovery_count"],
                job["status"],
            )
            tables[key].extend(job["results"])
    return tables


def render(tables: dict[tuple, list[dict]]) -> str:
    lines = []
    for key in sorted(tables, key=lambda k: (k[0], k[1], k[5], k[4])):
        suite, workload, step, lr, ratio, k, status = key
        records = tables[key]
        lines.append(
            f"### {suite}: {workload} (step {step}, lr {lr:g}) prune {ratio:g}, "
            f"K={k} [{status}]"
        )
        header = "| method | n | imm1 PPL | "
        header += "immK PPL | " if k > 1 else ""
        header += "final PPL | d_nc final | vs excp | vs insh |"
        lines.append(header)
        lines.append("|" + "---|" * (header.count("|") - 1))
        by_method: dict[str, list[dict]] = defaultdict(list)
        order: list[str] = []
        for record in records:
            if record["method"] not in by_method:
                order.append(record["method"])
            by_method[record["method"]].append(record)
        finals = {
            m: {r["seed"]: r["final"]["perplexity"] for r in rs} for m, rs in by_method.items()
        }
        control = finals.get("no_compression", {})
        for method in order:
            rs = by_method[method]
            imm1 = [r["cycles"][0]["after_recovery"]["perplexity"] for r in rs]
            immk = [r["cycles"][-1]["after_recovery"]["perplexity"] for r in rs]
            final = finals[method]
            shared = sorted(set(final) & set(control))
            d_nc = (
                f"{statistics.fmean(final[s] - control[s] for s in shared):+.3f}"
                if shared
                else "n/a"
            )
            row = f"| {method} | {len(rs)} | {_mean_std(imm1)} | "
            row += f"{_mean_std(immk)} | " if k > 1 else ""
            row += f"{_mean_std(list(final.values()))} | {d_nc} | "
            row += " | ".join(
                "-" if method == b else _sign(final, finals.get(b, {})) for b in BASELINES
            )
            lines.append(row + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = render(load_records(args.suites))
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
