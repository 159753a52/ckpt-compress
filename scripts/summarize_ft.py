#!/usr/bin/env python3
"""Summarize all fault-tolerant training experiments."""
import json, glob, os

results_dir = "results/paper_results/fault_tolerant"
configs = sorted(glob.glob(os.path.join(results_dir, "*_config.json")))

print(f"Found {len(configs)} FT experiments\n")
for f in configs:
    c = json.load(open(f))
    final = c.get('final', {})
    methods = list(final.keys())
    ts = os.path.basename(f).split('_ft_')[0]
    model_info = os.path.basename(f).replace('_config.json','').replace(ts+'_ft_','')
    
    print(f"=== {ts} | {model_info} ===")
    print(f"  K={c.get('num_recoveries','?')}, p={c.get('prune_ratio','?')}, lr={c.get('lr','?')}, schedule={c.get('lr_schedule','?')}")
    
    for m in methods:
        d = final[m]
        if 'perplexity' in d:
            print(f"  {m:40s} PPL={d['perplexity']:.2f}")
        elif 'accuracy' in d:
            print(f"  {m:40s} Acc={d['accuracy']:.4f}")
        else:
            print(f"  {m:40s} loss={d.get('loss','?')}")
    print()
