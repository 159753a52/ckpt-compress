## Reference

- smoke.json (commit f864e1b): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)

## Bytes vs restored perplexity

| method | operating point | n | eligible sparsity | restored PPL | zlib MB | lzma MB | torch.save MB | CR (fp32/zlib) | w / m / v zlib MB |
|---|---|---|---|---|---|---|---|---|---|
| excp_pipeline | alpha=0.001 | 1 | 0.8227 | 5.8907 +- 0.0000 | 102.72 | 98.78 | 608.7 | 47.4 | 78.7 / 12.3 / 11.8 |
| inshrinkerator_like | bins=16,prune=0.2,protect=0.005 | 1 | - | 733.8194 +- 0.0000 | 657.21 | 641.81 | 1353.5 | 7.4 | 271.0 / 228.6 / 157.6 |
| probe_topk_m1_exact_global_excpq | alpha=0.001 | 1 | 0.8227 | 5.8323 +- 0.0000 | 114.34 | 110.12 | 608.7 | 42.5 | 80.5 / 17.1 / 16.7 |

## Paired vs excp_pipeline (same alpha and seed; wins/ties/losses, lower is better)

| method | operating point | n | d PPL (mean) | PPL w/t/l | d zlib MB (mean) | bytes w/t/l |
|---|---|---|---|---|---|---|
| probe_topk_m1_exact_global_excpq | alpha=0.001 | 1 | -0.0584 | 1/0/0 | +11.63 | 0/0/1 |
