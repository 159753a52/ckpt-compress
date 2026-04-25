import json, os
from collections import defaultdict

results_dir = "results/paper_results"
entries = []
for dirpath, _, filenames in os.walk(results_dir):
    for fn in filenames:
        if fn.endswith("_config.json"):
            fp = os.path.join(dirpath, fn)
            try:
                with open(fp) as f:
                    data = json.load(f)
                model = data.get("model", "?")
                dataset = data.get("dataset", "?")
                prune = data.get("prune_ratio", 0.3)
                K = data.get("num_recoveries", 5)
                finals = data.get("final", {})
                for method, metrics in finals.items():
                    val = metrics.get("accuracy", metrics.get("perplexity", metrics.get("pearson", metrics.get("loss", 0))))
                    mk = "acc" if "accuracy" in metrics else ("ppl" if "perplexity" in metrics else "loss")
                    entries.append(f"{model:15s} {dataset:12s} K={K} p={prune:.1f} {method:35s} {mk}={val:.4f}")
            except Exception:
                pass

entries.sort()
for e in entries:
    print(e)
print(f"\nTotal: {len(entries)} entries")
