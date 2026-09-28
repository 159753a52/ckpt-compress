"""Emit LaTeX table rows (1e-3 nats) for the 32B sparsity sweep from raw JSON."""
import argparse
import hashlib
import json
from pathlib import Path

ROWS = [
    ("excp_style", "ExCP-style $|d|$"),
    ("inshrinkerator_style", "Inshrinkerator-style $|g\\theta|$"),
    ("first_order", "First order $|gd|$"),
    ("hvp_global", "Block HVP (global)"),
    ("dacp", "DACP (HVP + Weibull)"),
    ("probe_sample_m1", "Probe, sampled $m{=}1$"),
    ("probe_topk_m1", "Probe, top-$k$ $m{=}1$"),
    ("probe_headtail_m2", "Probe, head--tail $m{=}2$"),
    ("probe_headtail_m4", "Probe, head--tail $m{=}4$"),
    ("probe_topk_m8", "Probe, top-$k$ $m{=}8$"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sparsities", default="0.9,0.95,0.98,0.99,0.995")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = Path(args.input)
    r = json.loads(src.read_text())
    base = r["uncompressed"]["val"]
    sps = args.sparsities.split(",")
    lines = []
    best = {sp: min(r["evals"][k][sp]["val"] for k, _ in ROWS if k in r["evals"]) for sp in sps}
    for key, label in ROWS:
        if key not in r["evals"]:
            continue
        cells = []
        for sp in sps:
            v = r["evals"][key][sp]["val"]
            txt = f"{(v - base) * 1e3:+.1f}"
            cells.append(f"$\\mathbf{{{txt}}}$" if v == best[sp] else f"${txt}$")
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
        if key == "dacp":
            lines.append("\\midrule")
    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    prov = {"input": str(src), "input_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
