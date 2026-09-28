"""Summarize run_system_bytes_comparison.py outputs (all numbers read from JSON).

For each method and operating point: seeds, restored perplexity mean +- std,
total serialized bytes (torch.save, zlib level 6, lzma preset 1) mean over
seeds, compression ratio against the fp32 weights + both Adam moments, and
per-component zlib bytes. For every "*_excpq" method, paired comparisons
against excp_pipeline at the same alpha and seed are given as wins/ties/losses
for restored perplexity (lower wins) and for total zlib bytes (lower wins).

Usage: python summarize_system_bytes.py <json> [...] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _ms(values: list[float], digits: int = 4) -> str:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def _op_key(record: dict) -> str:
    op = record["operating_point"]
    if "alpha" in op:
        return f"alpha={op['alpha']:g}"
    return f"bins={op['n_bins']},prune={op['prune_fraction']:g},protect={op['protect_fraction']:g}"


def _wtl(pairs: list[tuple[float, float]]) -> str:
    wins = sum(a < b for a, b in pairs)
    ties = sum(a == b for a, b in pairs)
    return f"{wins}/{ties}/{len(pairs) - wins - ties}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    references = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        references.append((path.name, payload["reference"], payload["provenance"]["git_commit"]))
        for record in payload["records"]:
            groups[(record["method"], _op_key(record))].append(record)

    lines = ["## Reference", ""]
    for name, ref, commit in references:
        fp32 = ref["fp32_serialized_residual_and_moments"]["torch_save"]
        lines.append(
            f"- {name} (commit {commit[:7]}): checkpoint PPL {ref['checkpoint']['perplexity']:.4f}, "
            f"pretrained base PPL {ref['base']['perplexity']:.4f}, fp32 weights+exp_avg+exp_avg_sq "
            f"{fp32 / 1e6:.1f} MB ({ref['parameters']} parameters)"
        )
    fp32_total = references[0][1]["fp32_serialized_residual_and_moments"]["torch_save"]
    lines += [
        "",
        "## Bytes vs restored perplexity",
        "",
        "| method | operating point | n | eligible sparsity | restored PPL | zlib MB | lzma MB "
        "| torch.save MB | CR (fp32/zlib) | w / m / v zlib MB |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (method, op), records in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        zlib_total = [r["total_bytes"]["zlib6"] for r in records]
        lzma_total = [r["total_bytes"].get("lzma1", float("nan")) for r in records]
        raw_total = [r["total_bytes"]["torch_save"] for r in records]
        parts = " / ".join(
            f"{statistics.fmean(r['bytes'][p]['zlib6'] for r in records) / 1e6:.1f}"
            for p in ("weights", "exp_avg", "exp_avg_sq")
        )
        if records[0].get("excp_eligible_sparsity") is None:
            sparsity = "-"
        else:
            sparsity = f"{statistics.fmean(r['excp_eligible_sparsity'] for r in records):.4f}"
        lines.append(
            f"| {method} | {op} | {len(records)} | {sparsity} | "
            f"{_ms([r['restored']['perplexity'] for r in records])} | "
            f"{statistics.fmean(zlib_total) / 1e6:.2f} | {statistics.fmean(lzma_total) / 1e6:.2f} | "
            f"{statistics.fmean(raw_total) / 1e6:.1f} | {fp32_total / statistics.fmean(zlib_total):.1f} | {parts} |"
        )

    lines += [
        "",
        "## Paired vs excp_pipeline (same alpha and seed; wins/ties/losses, lower is better)",
        "",
        "| method | operating point | n | d PPL (mean) | PPL w/t/l | d zlib MB (mean) | bytes w/t/l |",
        "|---|---|---|---|---|---|---|",
    ]
    for (method, op), records in sorted(groups.items()):
        if not method.endswith("_excpq"):
            continue
        base = {r["seed"]: r for r in groups.get(("excp_pipeline", op), [])}
        pairs = [(r, base[r["seed"]]) for r in records if r["seed"] in base]
        if not pairs:
            continue
        ppl = [(a["restored"]["perplexity"], b["restored"]["perplexity"]) for a, b in pairs]
        size = [(a["total_bytes"]["zlib6"], b["total_bytes"]["zlib6"]) for a, b in pairs]
        lines.append(
            f"| {method} | {op} | {len(pairs)} | {statistics.fmean(a - b for a, b in ppl):+.4f} | "
            f"{_wtl(ppl)} | {statistics.fmean(a - b for a, b in size) / 1e6:+.2f} | {_wtl(size)} |"
        )
    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
