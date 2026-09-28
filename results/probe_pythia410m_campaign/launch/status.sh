#!/usr/bin/env bash
# Usage: status.sh [suite ...]  -- compact progress of the campaign
root=/home/gridtest/dacp_probe/campaign
grep "^\[" "$root/queue.log" 2>/dev/null | tail -4
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 7
df -h /home | tail -1
/home/gridtest/py310/bin/python - "$root" "$@" <<'EOF'
import json, sys, glob, os
root = sys.argv[1]
suites = sys.argv[2:] or sorted(os.listdir(root))
for s in suites:
    d = os.path.join(root, s)
    if not os.path.isdir(d):
        continue
    for f in sorted(glob.glob(os.path.join(d, "*_k*.json"))):
        try:
            j = json.load(open(f))
        except Exception as e:
            print(s, os.path.basename(f), "unreadable", e); continue
        c = j["config"]
        print(f"== {s} {c['name']} p={c['prune_ratio']} K={c['recovery_count']} lr={c['learning_rate']} status={j['status']} n={len(j['results'])}")
        if len(sys.argv) > 2:
            for r in j["results"]:
                cyc = r["cycles"]
                imm = cyc[0]["before_recovery"]["perplexity"]
                aft = [x["after_recovery"]["perplexity"] for x in cyc]
                print(f"   {r['method']:34s} s{r['seed']} imm={imm:8.3f} after={' '.join(f'{a:.3f}' for a in aft)} final={r['final']['perplexity']:.3f} wall={r['wall_seconds']:.0f}")
EOF

