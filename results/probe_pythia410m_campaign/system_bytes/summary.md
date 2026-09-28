## Reference

- excp_dacp_r2_s42.json (commit 0d39433): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- excp_dacp_r2_s43.json (commit 0d39433): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- excp_dacp_r2_s44.json (commit 0d39433): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- excp_dacp_s42.json (commit 0d39433): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- excp_dacp_s43.json (commit 0d39433): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- excp_dacp_s44.json (commit 0d39433): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- insh_p0_s42.json (commit 0d39433): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)

## Bytes vs restored perplexity

| method | operating point | n | eligible sparsity | restored PPL | zlib MB | lzma MB | torch.save MB | CR (fp32/zlib) | w / m / v zlib MB |
|---|---|---|---|---|---|---|---|---|---|
| dacp_excpq | alpha=0.0002 | 3 | 0.2374 | 5.5453 +- 0.0008 | 250.72 | 250.29 | 608.5 | 19.4 | 173.5 / 39.1 / 38.1 |
| dacp_excpq | alpha=0.0004 | 3 | 0.4492 | 5.5134 +- 0.0035 | 212.92 | 213.27 | 608.5 | 22.8 | 148.3 / 32.7 / 31.9 |
| dacp_excpq | alpha=0.0006 | 3 | 0.6197 | 5.5274 +- 0.0030 | 176.86 | 174.99 | 608.6 | 27.5 | 124.3 / 26.6 / 26.0 |
| dacp_excpq | alpha=0.001 | 3 | 0.8227 | 5.8635 +- 0.0085 | 114.66 | 110.30 | 608.7 | 42.4 | 80.7 / 17.2 / 16.8 |
| dacp_excpq | alpha=0.01 | 3 | 0.9018 | 6.6610 +- 0.0470 | 81.53 | 76.18 | 609.9 | 59.7 | 57.2 / 12.3 / 12.0 |
| dacp_excpq | alpha=1e-05 | 3 | 0.0121 | 5.5754 +- 0.0004 | 271.53 | 269.55 | 608.5 | 17.9 | 183.9 / 44.5 / 43.2 |
| dacp_excpq | alpha=5e-05 | 3 | 0.0604 | 5.5706 +- 0.0018 | 273.14 | 270.80 | 608.5 | 17.8 | 187.4 / 43.5 / 42.2 |
| excp_pipeline | alpha=0.0002 | 3 | 0.2374 | 5.5727 +- 0.0005 | 246.11 | 246.41 | 608.5 | 19.8 | 172.8 / 37.2 / 36.1 |
| excp_pipeline | alpha=0.0004 | 3 | 0.4492 | 5.5465 +- 0.0007 | 206.10 | 205.99 | 608.5 | 23.6 | 148.5 / 29.2 / 28.4 |
| excp_pipeline | alpha=0.0006 | 3 | 0.6197 | 5.5325 +- 0.0008 | 167.23 | 165.09 | 608.6 | 29.1 | 123.7 / 22.1 / 21.4 |
| excp_pipeline | alpha=0.001 | 3 | 0.8227 | 5.8906 +- 0.0005 | 102.92 | 98.92 | 608.7 | 47.3 | 78.9 / 12.2 / 11.8 |
| excp_pipeline | alpha=0.01 | 3 | 0.9018 | 6.5616 +- 0.0002 | 76.56 | 72.17 | 609.9 | 63.5 | 59.2 / 8.8 / 8.5 |
| excp_pipeline | alpha=1e-05 | 3 | 0.0121 | 5.5763 +- 0.0003 | 271.41 | 269.45 | 608.5 | 17.9 | 184.0 / 44.4 / 43.0 |
| excp_pipeline | alpha=5e-05 | 3 | 0.0604 | 5.5763 +- 0.0010 | 271.77 | 269.57 | 608.5 | 17.9 | 187.1 / 43.0 / 41.7 |
| inshrinkerator_like | bins=16,prune=0,protect=0.005 | 1 | - | 5.9669 +- 0.0000 | 673.71 | 653.61 | 1383.9 | 7.2 | 282.8 / 237.6 / 153.4 |
| inshrinkerator_like | bins=254,prune=0,protect=0.005 | 1 | - | 5.6010 +- 0.0000 | 1218.66 | 1193.41 | 1384.3 | 4.0 | 458.8 / 407.3 / 352.5 |
| inshrinkerator_like | bins=64,prune=0,protect=0.005 | 1 | - | 5.6175 +- 0.0000 | 933.26 | 964.57 | 1384.0 | 5.2 | 360.9 / 316.5 / 255.9 |
| probe_topk_m1_exact_global_excpq | alpha=0.0002 | 3 | 0.2374 | 5.5456 +- 0.0008 | 250.65 | 250.21 | 608.5 | 19.4 | 173.3 / 39.2 / 38.2 |
| probe_topk_m1_exact_global_excpq | alpha=0.0004 | 3 | 0.4492 | 5.5138 +- 0.0010 | 212.80 | 213.22 | 608.5 | 22.9 | 148.2 / 32.7 / 31.9 |
| probe_topk_m1_exact_global_excpq | alpha=0.0006 | 3 | 0.6197 | 5.5250 +- 0.0029 | 176.86 | 175.04 | 608.6 | 27.5 | 124.4 / 26.5 / 26.0 |
| probe_topk_m1_exact_global_excpq | alpha=0.001 | 3 | 0.8227 | 5.8496 +- 0.0158 | 114.73 | 110.43 | 608.7 | 42.4 | 80.8 / 17.1 / 16.8 |
| probe_topk_m1_exact_global_excpq | alpha=0.01 | 3 | 0.9018 | 6.6533 +- 0.0578 | 81.60 | 76.33 | 609.9 | 59.6 | 57.3 / 12.3 / 12.0 |
| probe_topk_m1_exact_global_excpq | alpha=1e-05 | 3 | 0.0121 | 5.5748 +- 0.0004 | 271.92 | 269.92 | 608.5 | 17.9 | 184.3 / 44.5 / 43.1 |
| probe_topk_m1_exact_global_excpq | alpha=5e-05 | 3 | 0.0604 | 5.5706 +- 0.0010 | 273.32 | 270.97 | 608.5 | 17.8 | 187.5 / 43.6 / 42.2 |

## Paired vs excp_pipeline (same alpha and seed; wins/ties/losses, lower is better)

| method | operating point | n | d PPL (mean) | PPL w/t/l | d zlib MB (mean) | bytes w/t/l |
|---|---|---|---|---|---|---|
| dacp_excpq | alpha=0.0002 | 3 | -0.0274 | 3/0/0 | +4.62 | 0/0/3 |
| dacp_excpq | alpha=0.0004 | 3 | -0.0331 | 3/0/0 | +6.82 | 0/0/3 |
| dacp_excpq | alpha=0.0006 | 3 | -0.0051 | 3/0/0 | +9.63 | 0/0/3 |
| dacp_excpq | alpha=0.001 | 3 | -0.0270 | 3/0/0 | +11.74 | 0/0/3 |
| dacp_excpq | alpha=0.01 | 3 | +0.0994 | 0/0/3 | +4.97 | 0/0/3 |
| dacp_excpq | alpha=1e-05 | 3 | -0.0009 | 3/0/0 | +0.12 | 1/0/2 |
| dacp_excpq | alpha=5e-05 | 3 | -0.0057 | 3/0/0 | +1.37 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=0.0002 | 3 | -0.0271 | 3/0/0 | +4.54 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=0.0004 | 3 | -0.0327 | 3/0/0 | +6.70 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=0.0006 | 3 | -0.0075 | 3/0/0 | +9.63 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=0.001 | 3 | -0.0410 | 3/0/0 | +11.80 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=0.01 | 3 | +0.0917 | 0/0/3 | +5.03 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=1e-05 | 3 | -0.0016 | 3/0/0 | +0.51 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=5e-05 | 3 | -0.0057 | 3/0/0 | +1.56 | 0/0/3 |
