"""Summarise probe_small_models job JSONs: immediate and final metric per method.

Usage: python summarize_probe_small_models.py <result_dir> [metric]

"Immediate" is the metric right after the first lossy restoration
(cycles[0].after_recovery); "final" is the metric after the last recovery
segment. Values are averaged over the seeds present for each
(prune_ratio, method); the delta columns are relative to no_compression on
the same seeds.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def iter_records(node):
    if isinstance(node, dict):
        if {"method", "seed", "cycles", "final", "prune_ratio"} <= node.keys():
            yield node
            return
        for value in node.values():
            yield from iter_records(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_records(value)


def main() -> None:
    root = Path(sys.argv[1])
    metric = sys.argv[2] if len(sys.argv) > 2 else "perplexity"
    table = defaultdict(dict)
    for path in sorted(root.glob("*.json")):
        if path.name in {"suite_manifest.json", "plan.json"}:
            continue
        for record in iter_records(json.loads(path.read_text(encoding="utf-8"))):
            cycles = record["cycles"]
            immediate = cycles[0]["after_recovery"][metric] if cycles else None
            key = (float(record["prune_ratio"]), record["method"])
            table[key][int(record["seed"])] = (immediate, record["final"][metric])
    methods = []
    for _, method in table:
        if method not in methods:
            methods.append(method)
    for ratio in sorted({ratio for ratio, _ in table}):
        base = table.get((ratio, "no_compression"), {})
        print(f"\nprune_ratio={ratio}  metric={metric}")
        print(f"{'method':34s} {'seeds':10s} {'immediate':>10s} {'final':>10s} "
              f"{'d_imm':>9s} {'d_final':>9s}")
        for method in methods:
            runs = table.get((ratio, method))
            if not runs:
                continue
            seeds = sorted(runs)
            imm = statistics.mean(runs[s][0] for s in seeds)
            fin = statistics.mean(runs[s][1] for s in seeds)
            paired = [s for s in seeds if s in base]
            if paired:
                d_imm = statistics.mean(runs[s][0] - base[s][0] for s in paired)
                d_fin = statistics.mean(runs[s][1] - base[s][1] for s in paired)
                deltas = f"{d_imm:9.4f} {d_fin:9.4f}"
            else:
                deltas = f"{'n/a':>9s} {'n/a':>9s}"
            seed_text = ",".join(str(s) for s in seeds)
            print(f"{method:34s} {seed_text:10s} {imm:10.4f} {fin:10.4f} {deltas}")


if __name__ == "__main__":
    main()
