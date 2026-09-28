#!/usr/bin/env bash
# Usage: sysstat.sh <name>  -- compact view of a system_bytes JSON
/home/gridtest/py310/bin/python - "/home/gridtest/dacp_probe/campaign/system_bytes/$1.json" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
r = j["reference"]
print("ckpt ppl", round(r["checkpoint"]["perplexity"], 4), "base ppl", round(r["base"]["perplexity"], 4), "fp32 total", r["fp32_serialized_residual_and_moments"]["torch_save"], "elapsed", round(j["elapsed_seconds"]))
for x in j["records"]:
    tb = x["total_bytes"]; b = x["bytes"]
    op = x["operating_point"]
    print(f"{x['method']:32s} s{x['seed']} op={op.get('alpha', op.get('n_bins'))} sp={x.get('excp_eligible_sparsity', float('nan')):.4f} "
          f"keptE={x.get('eligible_kept','-')} ppl={x['restored']['perplexity']:.4f} "
          f"zlib={tb['zlib6']/1e6:.2f}MB lzma={tb.get('lzma1', 0)/1e6:.2f}MB raw={tb['torch_save']/1e6:.1f}MB "
          f"w/m/v zlib={b['weights']['zlib6']/1e6:.1f}/{b['exp_avg']['zlib6']/1e6:.1f}/{b['exp_avg_sq']['zlib6']/1e6:.1f} "
          f"err={ {k: round(v, 3) for k, v in x['relative_l2_error'].items()} } t={x['seconds']:.0f}s "
          f"{(x.get('mask_method') or {}).get('scoring_seconds', '')}")
EOF
