"""Readers for the raw result records.

Each loader returns ``None`` (or an empty container) when its files are
missing, and adds every file it reads to ``inputs`` so that the figure's
provenance lists the exact bytes used.  Only fields written by the producing
scripts are accessed:

* sweep/{step100,base}.json  - experiments/scripts/run_probe_sweep_32b.py
* cycle1/branches/*.json, chains/*.json, cycle1/score_records.json
                              - experiments/scripts/run_probe_coverage_32b.py
* system_bytes/*.json        - run_system_bytes_comparison.py
                              (schema as read by summarize_system_bytes.py)
* main_k1/*_k*.json          - run_paper_experiments.py job records
                              (schema as read by summarize_probe_pythia410m_campaign.py)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set


def _load(path: Path, inputs: Set[Path]):
    path = Path(path)
    inputs.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------- 32B sweep
def load_sweep(run_dir: Path, inputs: Set[Path]) -> Dict[str, dict]:
    """{'step100': res, 'base': res} for the files that exist."""
    out = {}
    for regime in ("step100", "base"):
        p = Path(run_dir) / "sweep" / f"{regime}.json"
        if p.exists():
            out[regime] = _load(p, inputs)
    return out


# --------------------------------------------------------- 32B single cycle
def load_branches(run_dir: Path, inputs: Set[Path], ids: List[str]) -> Dict[str, dict]:
    d = Path(run_dir) / "cycle1" / "branches"
    return {i: _load(d / f"{i}.json", inputs) for i in ids if (d / f"{i}.json").exists()}


def load_chains(run_dir: Path, inputs: Set[Path]) -> Dict[str, dict]:
    d = Path(run_dir) / "chains"
    if not d.exists():
        return {}
    return {p.stem: _load(p, inputs) for p in sorted(d.glob("*.json"))}


def load_score_records(run_dir: Path, inputs: Set[Path]) -> Optional[dict]:
    p = Path(run_dir) / "cycle1" / "score_records.json"
    return _load(p, inputs) if p.exists() else None


def peak_max(peak: dict) -> float:
    """Largest per-device peak (GiB) from a {'cuda:i': GiB} record."""
    return max(peak.values())


# ------------------------------------------------------ Pythia system bytes
SYSTEM_BYTES_SKIP = {"smoke.json"}  # smoke test, not part of the campaign summary


def load_system_bytes(sb_dir: Path, inputs: Set[Path]):
    """(records, reference) from every system_bytes JSON except the smoke run."""
    d = Path(sb_dir)
    if not d.exists():
        return [], None
    records, reference = [], None
    for p in sorted(d.glob("*.json")):
        if p.name in SYSTEM_BYTES_SKIP:
            continue
        payload = _load(p, inputs)
        reference = reference or payload["reference"]
        for r in payload["records"]:
            records.append(dict(r, _file=p.name))
    return records, reference


# ------------------------------------------------------- Pythia main_k1 jobs
def load_jobs(suite_dir: Path, inputs: Set[Path]) -> Dict[float, Dict[str, Dict[int, dict]]]:
    """{prune_ratio: {method: {seed: result}}} from complete job JSONs."""
    d = Path(suite_dir)
    out: Dict[float, Dict[str, Dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    if not d.exists():
        return {}
    for p in sorted(d.glob("*_k*.json")):
        job = _load(p, inputs)
        if job.get("status") != "complete":
            continue
        ratio = float(job["config"]["prune_ratio"])
        for r in job["results"]:
            out[ratio][r["method"]][int(r["seed"])] = r
    return {k: dict(v) for k, v in out.items()}
