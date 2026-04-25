#!/usr/bin/env python3
"""Print table1 and FT results in a readable format."""
import json, glob, os

base = "results/paper_results"

# Table1 single-shot results
print("="*80)
print("TABLE 1 (Single-shot compression)")
print("="*80)
for f in sorted(glob.glob(os.path.join(base, "table1", "*_table1_*.json"))):
    if "_config" in f:
        continue
    name = os.path.basename(f)
    print(f"\n--- {name} ---")
    data = json.load(open(f))
    baseline_ppl = None
    for r in data["results"]:
        if baseline_ppl is None:
            baseline_ppl = r.get("baseline_ppl", "?")
        method = r["method"]
        ratio = r["target_ratio"]
        if "perplexity" in r:
            metric = f"PPL={r['perplexity']:.2f}"
        elif "accuracy" in r:
            metric = f"Acc={r['accuracy']:.4f}"
        else:
            metric = f"loss={r['loss']:.4f}"
        print(f"  {method:40s}  sparsity={ratio:.0%}  {metric}")
    if baseline_ppl:
        print(f"  [Baseline PPL/Acc: {baseline_ppl}]")

# FT results
print("\n" + "="*80)
print("FAULT-TOLERANT TRAINING RESULTS")
print("="*80)
for f in sorted(glob.glob(os.path.join(base, "fault_tolerant", "*_ft_*.json"))):
    if "_config" in f:
        continue
    name = os.path.basename(f)
    print(f"\n--- {name} ---")
    data = json.load(open(f))
    # Check structure
    if "results" in data and isinstance(data["results"], dict):
        for method_name, method_data in data["results"].items():
            final = method_data.get("final_metrics", {})
            if "perplexity" in final:
                metric = f"PPL={final['perplexity']:.2f}"
            elif "accuracy" in final:
                metric = f"Acc={final['accuracy']:.4f}"
            else:
                metric = f"loss={final.get('loss', '?')}"
            print(f"  {method_name:40s}  {metric}")
    elif isinstance(data, list):
        # Flat list format
        methods = {}
        for r in data:
            m = r.get("method", "?")
            if m not in methods:
                methods[m] = []
            methods[m].append(r)
        for m, records in methods.items():
            last = records[-1]
            ppl = last.get("perplexity", 0)
            acc = last.get("accuracy", 0)
            loss = last.get("loss", 0)
            print(f"  {m:40s}  final: PPL={ppl:.2f} Acc={acc:.4f} loss={loss:.4f}")
    elif isinstance(data, dict):
        for k, v in data.items():
            if k != "config":
                print(f"  {k}: {v}")
