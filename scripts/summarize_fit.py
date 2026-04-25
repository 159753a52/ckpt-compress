#!/usr/bin/env python3
"""Summarize distribution fit results from gamma_validation JSON."""
import json, math, sys, glob

results_dir = "results/paper_results/gamma_validation"
fit_files = sorted(glob.glob(f"{results_dir}/*_fit_results.json"))

for fpath in fit_files:
    print(f"\n{'='*60}")
    print(f"File: {fpath}")
    d = json.load(open(fpath))
    print(f"  Total layers: {len(d)}")
    for dn in ["gamma", "lognormal", "weibull", "exponential"]:
        key = f"{dn}_ks"
        vals = [r[key] for r in d if not math.isnan(r.get(key, float("nan")))]
        if vals:
            print(f"  {dn:12s}: avg_ks={sum(vals)/len(vals):.4f}, n={len(vals)}")
    # gamma pass rate
    gp = sum(1 for r in d if r.get("gamma_pvalue", 0) > 0.05)
    print(f"  gamma_pass: {gp}/{len(d)} ({gp/max(len(d),1)*100:.1f}%)")
