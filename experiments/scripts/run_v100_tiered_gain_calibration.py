"""Calibrate a residual gain restricted to slow coordinates, chosen by calibration loss only.

The gain multiplies the retained eligible residual values at coordinates whose stored Adam second
moment lies below a quantile of its eligible values; the other coordinates keep gain 1. The scope
quantile and the gain are chosen together by the lowest loss on the 64 training-calibration batches:
a coarse grid for every scope, then a 0.05 grid around the best coarse point. Scope 1 is the
uniform gain. Validation batches are not used.
"""
from pathlib import Path
import argparse
import json
import math
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--encoded-dir", type=Path, required=True)
    ap.add_argument("--method", choices=("iterative_ucb_disjoint", "excp_style", "inshrinkerator_style"), required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scopes", default="0.25,0.5,1")
    ap.add_argument("--gains", default=",".join(f"{1 + 0.25 * i:g}" for i in range(15)))
    ap.add_argument("--ppl-gate", type=float, default=.5)
    args = ap.parse_args()
    import torch
    from run_residual_sweep_small import REGIMES, load_regime
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, sha256_file, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from experiments.lib.calibrated_archive import load_payload
    from run_system_bytes_comparison import excp_dequant
    from transformers import GPTNeoXForCausalLM
    scopes = [float(x) for x in args.scopes.split(",")]
    coarse = [float(x) for x in args.gains.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = json.loads((args.encoded_dir / "completed.json").read_text())["selected"][args.method]
    rec_path = args.encoded_dir / selected["file"]
    if sha256_file(rec_path) != selected["sha256"]:
        raise RuntimeError("source archive record hash differs")
    rec = json.loads(rec_path.read_text())
    archive = args.encoded_dir / rec["archive"]["file"]
    if sha256_file(archive) != rec["archive"]["sha256"]:
        raise RuntimeError("source archive hash differs")
    identity = {"checkpoint_sha256": sha256_file(args.checkpoint), "source_archive_sha256": sha256_file(archive),
                "script_sha256": sha256_file(Path(__file__)), "scopes": scopes, "coarse_gains": coarse, "seed": 42,
                "selection": "lowest calibration loss over scope and gain; coarse grid, then 0.05 steps around the best coarse point",
                "ppl_gate_percent": args.ppl_gate}
    if rec["identity"]["checkpoint_sha256"] != identity["checkpoint_sha256"]:
        raise RuntimeError("source checkpoint differs")
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    s, pool = ctx["sweep"], ctx["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    offset = len(plan.training_segments[0])
    used = set(plan.selected_pool_indices[offset:offset + 16])
    gate = [pool[i] for i in range(len(pool)) if i not in used][:64]
    baseline_gate = evaluate_lm(s.model, gate, "cuda:0")
    identity["gate_sha256"] = batch_hash(gate)
    identity["gain_parameter_names"] = s.names
    payload = load_payload(archive)
    base_model = GPTNeoXForCausalLM.from_pretrained(args.data_root / "models/pythia-410m", dtype=torch.float32, local_files_only=True)
    base = {n: p.detach().cpu() for n, p in base_model.named_parameters()}
    residual = {n: excp_dequant(value, base[n].shape, "cpu").float() for n, value in payload["model"].items()}
    v = {n: excp_dequant(payload["exp_avg_sq"][n], base[n].shape, "cpu").float() for n in s.names}
    buffers = payload["buffers"]
    del payload
    every = torch.cat([v[n].flatten() for n in s.names])
    sample = every[torch.randint(0, every.numel(), (4_000_000,), generator=torch.Generator().manual_seed(0))]
    thresholds = {q: (float(torch.quantile(sample, q)) if q < 1 else math.inf) for q in scopes}
    del every, sample
    identity["v_thresholds"] = {f"{q:g}": t for q, t in thresholds.items()}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    points = {}

    def measure(q, gain):
        key = f"q{q:g}_g{gain:g}"
        if key in points:
            return points[key]
        path = args.out_dir / f"gate_{key}.json"
        if path.exists():
            point = json.loads(path.read_text())
            if point["identity"] != identity:
                raise RuntimeError("existing tiered calibration differs")
        else:
            tiers = {"v_quantile": q, "v_threshold": thresholds[q], "slow": gain, "fast": 1.}
            with torch.no_grad():
                for n, p in s.model.named_parameters():
                    r = residual[n]
                    if n in v:
                        r = torch.where(v[n] < tiers["v_threshold"], r * tiers["slow"], r * tiers["fast"])
                    p.copy_(base[n].to(s.device) + r.to(s.device))
            s.model.load_state_dict({**s.model.state_dict(), **buffers}, strict=True)
            quality = evaluate_lm(s.model, gate, s.device)
            change = 100 * math.expm1(quality["loss"] - baseline_gate["loss"])
            point = {"identity": identity, "gain": gain, "tiered_gain": tiers, "gate": quality,
                     "gate_ppl_change_percent": change, "passed_gate": change <= args.ppl_gate, "code_commit": commit}
            write_json(path, point)
        points[key] = point
        print(f"GATE {args.method} scope={q:g} gain={gain:g} change={point['gate_ppl_change_percent']:.5f}", flush=True)
        return point

    for q in scopes:
        for gain in coarse:
            measure(q, gain)
    best = min(points.values(), key=lambda p: p["gate"]["loss"])
    q0, g0 = best["tiered_gain"]["v_quantile"], best["gain"]
    for i in range(-4, 5):
        if g0 + 0.05 * i >= 1:
            measure(q0, round(g0 + 0.05 * i, 2))
    best = min(points.values(), key=lambda p: p["gate"]["loss"])
    write_json(args.out_dir / "completed.json", {"status": "completed", "identity": identity,
               "source_record": str(rec_path), "source_record_sha256": sha256_file(rec_path), "method": args.method,
               "points": points, "selected": best, "selected_by": "lowest calibration loss over scope and gain",
               "code_commit": commit, "scope": "calibration only; no main-subset or independent validation result used"})
    print("SELECTED " + json.dumps({k: best[k] for k in ("gain", "tiered_gain", "gate_ppl_change_percent")}), flush=True)


if __name__ == "__main__":
    main()
