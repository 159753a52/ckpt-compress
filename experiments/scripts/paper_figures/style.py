"""Shared figure style, method registry and provenance-aware saving.

Every method keeps one colour and one marker in all figures.  Main methods use
the Okabe-Ito palette plus three Tol colours for the probe estimators; other
probe variants reuse their family colour mixed with white.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
from PIL import Image  # noqa: E402  (matplotlib dependency)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402

SINGLE_W = 3.3  # in, one column
DOUBLE_W = 6.9  # in, full text width
PREVIEW_MAX_PX = 1800


def lighten(color: str, amount: float) -> str:
    """Mix ``color`` with white; amount=0 keeps it, 1 gives white."""
    r, g, b = to_rgb(color)
    return to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


C = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "blue": "#0072B2",
    "rose": "#CC79A7",
    "wine": "#882255",
    "indigo": "#332288",
}


@dataclass(frozen=True)
class MethodStyle:
    label: str
    color: str
    marker: str
    ls: str = "-"
    main: bool = True


METHODS: Dict[str, MethodStyle] = {
    "no_compression": MethodStyle("No compression", C["black"], "", "-"),
    "excp_style": MethodStyle("ExCP-style", C["orange"], "s"),
    "inshrinkerator_style": MethodStyle("Inshrinkerator-style", C["sky"], "^"),
    "first_order": MethodStyle("First-order", C["green"], "D"),
    "hvp_global": MethodStyle("Block HVP (global)", C["vermillion"], "o"),
    "dacp": MethodStyle("DACP", C["blue"], "P"),
    "probe_topk_m1": MethodStyle("Probe top-$k$ $m{=}1$", C["rose"], "v"),
    "probe_headtail_m2": MethodStyle("Probe head-tail $m{=}2$", C["wine"], ">"),
    "probe_sample_m1": MethodStyle("Probe sampled $m{=}1$", C["indigo"], "<"),
    # secondary probe variants: lighter shade of the family colour
    "probe_topk_m2": MethodStyle("Probe top-$k$ $m{=}2$", lighten(C["rose"], 0.45), "v", "-", False),
    "probe_topk_m4": MethodStyle("Probe top-$k$ $m{=}4$", lighten(C["rose"], 0.55), "v", "-", False),
    "probe_topk_m8": MethodStyle("Probe top-$k$ $m{=}8$", lighten(C["rose"], 0.65), "v", "-", False),
    "probe_headtail_m4": MethodStyle("Probe head-tail $m{=}4$", lighten(C["wine"], 0.5), ">", "-", False),
    "probe_sample_m4": MethodStyle("Probe sampled $m{=}4$", lighten(C["indigo"], 0.5), "<", "-", False),
    "probe_mean95": MethodStyle("Probe sampled (mean-mass rule)", lighten(C["indigo"], 0.35), "<", "-", False),
}

ALIASES = {
    "hvp": "hvp_global",
    "taylor_exact_global": "hvp_global",
    "first_order_exact_global": "first_order",
    "excp_pipeline": "excp_style",
    "inshrinkerator_like": "inshrinkerator_style",
}

_PROBE_SEED = re.compile(r"^probe_s\d+_m(\d+)_global(?:_[0-9.]+)?$")


def canon(name: str) -> str:
    """Map a raw method/tag/branch name onto a METHODS key."""
    n = re.sub(r"_0\.\d+$", "", name)  # branch/chain ids carry a trailing sparsity
    if n.endswith("_excpq"):
        n = n[: -len("_excpq")]
    if n in ALIASES:
        return ALIASES[n]
    m = _PROBE_SEED.match(n)
    if m:  # 32B single-cycle probes are sampled probes with m rounds
        return f"probe_sample_m{m.group(1)}"
    if n.endswith("_exact_global"):
        n = n[: -len("_exact_global")]
    return ALIASES.get(n, n)


def style_of(name: str) -> MethodStyle:
    key = canon(name)
    if key in METHODS:
        return METHODS[key]
    fam = "indigo" if "sample" in key else "wine" if "headtail" in key else "rose"
    if key.startswith("probe"):
        return MethodStyle(key.replace("_", " "), lighten(C[fam], 0.6), "v", "-", False)
    return MethodStyle(key.replace("_", " "), "#8C8C8C", "x", "-", False)


def line_kw(name: str, marker: bool = True, **over) -> dict:
    st = style_of(name)
    kw = dict(color=st.color, ls=st.ls, label=st.label, lw=1.1 if st.main else 0.8, zorder=3 if st.main else 2)
    if marker and st.marker:
        kw.update(marker=st.marker, ms=3.8 if st.main else 2.6, mec=st.color, mfc=st.color, mew=0.6)
    kw.update(over)
    return kw


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.columnspacing": 1.0,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "lines.linewidth": 1.1,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 100,
        }
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_preview(fig, path: Path) -> tuple:
    """PNG preview whose width and height are both <= PREVIEW_MAX_PX."""
    w, h = fig.get_size_inches()
    dpi = min(300, int((PREVIEW_MAX_PX - 20) / max(w, h)))
    for _ in range(6):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
        with Image.open(path) as im:
            pw, ph = im.size
        if max(pw, ph) <= PREVIEW_MAX_PX:
            return pw, ph, dpi
        dpi = int(math.floor(dpi * (PREVIEW_MAX_PX - 10) / max(pw, ph)))
    raise RuntimeError(f"cannot fit preview {path} into {PREVIEW_MAX_PX}px")


def save(fig, out_dir: Path, stem: str, inputs: Iterable[Path], notes: Optional[dict] = None) -> Dict[str, Path]:
    """Write <stem>.pdf, <stem>.png (preview) and <stem>.provenance.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"pdf": out_dir / f"{stem}.pdf", "png": out_dir / f"{stem}.png"}
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    pw, ph, dpi = _save_preview(fig, paths["png"])
    plt.close(fig)
    here = Path(__file__).resolve().parent
    prov = {
        "figure": stem,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python " + " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "scripts": {p.name: sha256(p) for p in sorted(here.glob("*.py"))},
        "inputs": {str(p): sha256(p) for p in sorted({Path(p).resolve() for p in inputs})},
        "outputs": {p.name: sha256(p) for p in paths.values()},
        "preview_png": {"width_px": pw, "height_px": ph, "dpi": dpi},
        "matplotlib": matplotlib.__version__,
    }
    if notes:
        prov["notes"] = notes
    (out_dir / f"{stem}.provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return paths


def moving_average(xs, w: int = 5):
    """Trailing w-point mean (window shrinks at the start)."""
    out = []
    for i in range(len(xs)):
        seg = xs[max(0, i - w + 1): i + 1]
        out.append(sum(seg) / len(seg))
    return out
