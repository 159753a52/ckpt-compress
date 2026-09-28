## Reference

- excp_dacp_s42.json (commit 0d39433): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- excp_dacp_s43.json (commit 0d39433): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- excp_dacp_s44.json (commit 0d39433): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- excp_dacp_r2_s42.json (commit 0d39433): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- excp_dacp_r2_s43.json (commit 0d39433): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- excp_dacp_r2_s44.json (commit 0d39433): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- iter_R1_alphas_s42.json (commit 9f2d62f): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- iter_R1_alphas_s43.json (commit 9f2d62f): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- iter_R1_alphas_s44.json (commit 9f2d62f): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- iter_R1_beyond_s42.json (commit 9f2d62f): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- iter_R1_beyond_s43.json (commit 9f2d62f): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- iter_R1_beyond_s44.json (commit 9f2d62f): checkpoint PPL 5.5910, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)

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
| first_order_excpq | alpha=0.01,eligible_sp=0.95 (beyond ExCP cap) | 3 | 0.9500 | 7.9810 +- 0.0444 | 58.48 | 53.02 | 609.9 | 83.2 | 41.5 / 8.6 / 8.3 |
| first_order_excpq | alpha=0.01,eligible_sp=0.98 (beyond ExCP cap) | 3 | 0.9800 | 9.8141 +- 0.0733 | 39.89 | 35.53 | 609.9 | 122.0 | 29.0 / 5.5 / 5.3 |
| first_order_excpq | alpha=0.01,eligible_sp=0.99 (beyond ExCP cap) | 3 | 0.9900 | 10.9033 +- 0.0679 | 31.91 | 28.73 | 609.9 | 152.4 | 23.8 / 4.2 / 3.9 |
| iterative_first_order_excpq | alpha=0.0002 | 3 | 0.2374 | 5.5456 +- 0.0009 | 250.72 | 250.27 | 608.5 | 19.4 | 173.4 / 39.2 / 38.1 |
| iterative_first_order_excpq | alpha=0.0004 | 3 | 0.4492 | 5.5132 +- 0.0026 | 212.75 | 213.16 | 608.5 | 22.9 | 148.1 / 32.7 / 31.9 |
| iterative_first_order_excpq | alpha=0.0006 | 3 | 0.6197 | 5.5276 +- 0.0023 | 176.88 | 175.11 | 608.6 | 27.5 | 124.6 / 26.4 / 25.9 |
| iterative_first_order_excpq | alpha=0.001 | 3 | 0.8227 | 6.0015 +- 0.0239 | 114.55 | 110.28 | 608.7 | 42.5 | 81.0 / 17.0 / 16.6 |
| iterative_first_order_excpq | alpha=0.01 | 3 | 0.9018 | 6.6444 +- 0.0522 | 81.54 | 76.26 | 609.9 | 59.6 | 57.4 / 12.2 / 11.9 |
| iterative_first_order_excpq | alpha=0.01,eligible_sp=0.95 (beyond ExCP cap) | 3 | 0.9500 | 7.3470 +- 0.0820 | 58.37 | 52.87 | 609.9 | 83.3 | 41.6 / 8.5 / 8.3 |
| iterative_first_order_excpq | alpha=0.01,eligible_sp=0.98 (beyond ExCP cap) | 3 | 0.9800 | 8.0835 +- 0.1052 | 39.63 | 35.29 | 609.9 | 122.7 | 28.9 / 5.5 / 5.3 |
| iterative_first_order_excpq | alpha=0.01,eligible_sp=0.99 (beyond ExCP cap) | 3 | 0.9900 | 8.7559 +- 0.0617 | 31.63 | 28.50 | 609.9 | 153.8 | 23.6 / 4.1 / 3.9 |
| iterative_first_order_excpq | alpha=1e-05 | 3 | 0.0121 | 5.5752 +- 0.0006 | 271.49 | 269.54 | 608.5 | 17.9 | 183.8 / 44.5 / 43.2 |
| iterative_first_order_excpq | alpha=5e-05 | 3 | 0.0604 | 5.5709 +- 0.0006 | 273.27 | 270.93 | 608.5 | 17.8 | 187.5 / 43.5 / 42.2 |
| iterative_ucb_excpq | alpha=0.0002 | 3 | 0.2374 | 5.5732 +- 0.0008 | 250.23 | 249.43 | 608.5 | 19.4 | 172.0 / 39.6 / 38.6 |
| iterative_ucb_excpq | alpha=0.0004 | 3 | 0.4492 | 5.5500 +- 0.0024 | 212.36 | 212.50 | 608.5 | 22.9 | 146.4 / 33.4 / 32.6 |
| iterative_ucb_excpq | alpha=0.0006 | 3 | 0.6197 | 5.5288 +- 0.0010 | 176.69 | 174.74 | 608.6 | 27.5 | 122.7 / 27.3 / 26.7 |
| iterative_ucb_excpq | alpha=0.001 | 3 | 0.8227 | 5.6981 +- 0.0058 | 114.87 | 110.42 | 608.7 | 42.3 | 79.7 / 17.8 / 17.4 |
| iterative_ucb_excpq | alpha=0.01 | 3 | 0.9018 | 6.2380 +- 0.0246 | 81.88 | 76.40 | 609.9 | 59.4 | 56.4 / 12.9 / 12.5 |
| iterative_ucb_excpq | alpha=0.01,eligible_sp=0.95 (beyond ExCP cap) | 3 | 0.9500 | 6.9297 +- 0.0704 | 58.81 | 53.09 | 609.9 | 82.7 | 40.8 / 9.2 / 8.8 |
| iterative_ucb_excpq | alpha=0.01,eligible_sp=0.98 (beyond ExCP cap) | 3 | 0.9800 | 7.8415 +- 0.0592 | 39.95 | 35.41 | 609.9 | 121.8 | 28.3 / 6.0 / 5.7 |
| iterative_ucb_excpq | alpha=0.01,eligible_sp=0.99 (beyond ExCP cap) | 3 | 0.9900 | 8.6636 +- 0.0462 | 31.73 | 28.48 | 609.9 | 153.3 | 23.1 / 4.4 / 4.2 |
| iterative_ucb_excpq | alpha=1e-05 | 3 | 0.0121 | 5.5757 +- 0.0002 | 271.58 | 269.61 | 608.5 | 17.9 | 183.9 / 44.5 / 43.2 |
| iterative_ucb_excpq | alpha=5e-05 | 3 | 0.0604 | 5.5763 +- 0.0016 | 272.58 | 270.31 | 608.5 | 17.8 | 186.5 / 43.7 / 42.4 |
| probe_topk_m1_exact_global_excpq | alpha=0.0002 | 3 | 0.2374 | 5.5456 +- 0.0008 | 250.65 | 250.21 | 608.5 | 19.4 | 173.3 / 39.2 / 38.2 |
| probe_topk_m1_exact_global_excpq | alpha=0.0004 | 3 | 0.4492 | 5.5138 +- 0.0010 | 212.80 | 213.22 | 608.5 | 22.9 | 148.2 / 32.7 / 31.9 |
| probe_topk_m1_exact_global_excpq | alpha=0.0006 | 3 | 0.6197 | 5.5250 +- 0.0029 | 176.86 | 175.04 | 608.6 | 27.5 | 124.4 / 26.5 / 26.0 |
| probe_topk_m1_exact_global_excpq | alpha=0.001 | 3 | 0.8227 | 5.8496 +- 0.0158 | 114.73 | 110.43 | 608.7 | 42.4 | 80.8 / 17.1 / 16.8 |
| probe_topk_m1_exact_global_excpq | alpha=0.01 | 3 | 0.9018 | 6.6533 +- 0.0578 | 81.60 | 76.33 | 609.9 | 59.6 | 57.3 / 12.3 / 12.0 |
| probe_topk_m1_exact_global_excpq | alpha=0.01,eligible_sp=0.95 (beyond ExCP cap) | 3 | 0.9500 | 8.0263 +- 0.0496 | 58.49 | 53.03 | 609.9 | 83.2 | 41.6 / 8.6 / 8.3 |
| probe_topk_m1_exact_global_excpq | alpha=0.01,eligible_sp=0.98 (beyond ExCP cap) | 3 | 0.9800 | 9.8472 +- 0.0363 | 39.89 | 35.53 | 609.9 | 121.9 | 29.1 / 5.5 / 5.3 |
| probe_topk_m1_exact_global_excpq | alpha=0.01,eligible_sp=0.99 (beyond ExCP cap) | 3 | 0.9900 | 10.9279 +- 0.0286 | 31.90 | 28.73 | 609.9 | 152.5 | 23.8 / 4.2 / 3.9 |
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
| iterative_first_order_excpq | alpha=0.0002 | 3 | -0.0271 | 3/0/0 | +4.61 | 0/0/3 |
| iterative_first_order_excpq | alpha=0.0004 | 3 | -0.0334 | 3/0/0 | +6.65 | 0/0/3 |
| iterative_first_order_excpq | alpha=0.0006 | 3 | -0.0049 | 3/0/0 | +9.65 | 0/0/3 |
| iterative_first_order_excpq | alpha=0.001 | 3 | +0.1109 | 0/0/3 | +11.63 | 0/0/3 |
| iterative_first_order_excpq | alpha=0.01 | 3 | +0.0828 | 0/0/3 | +4.98 | 0/0/3 |
| iterative_first_order_excpq | alpha=1e-05 | 3 | -0.0011 | 3/0/0 | +0.08 | 1/0/2 |
| iterative_first_order_excpq | alpha=5e-05 | 3 | -0.0054 | 3/0/0 | +1.51 | 0/0/3 |
| iterative_ucb_excpq | alpha=0.0002 | 3 | +0.0005 | 1/0/2 | +4.12 | 0/0/3 |
| iterative_ucb_excpq | alpha=0.0004 | 3 | +0.0034 | 0/0/3 | +6.26 | 0/0/3 |
| iterative_ucb_excpq | alpha=0.0006 | 3 | -0.0037 | 3/0/0 | +9.45 | 0/0/3 |
| iterative_ucb_excpq | alpha=0.001 | 3 | -0.1925 | 3/0/0 | +11.94 | 0/0/3 |
| iterative_ucb_excpq | alpha=0.01 | 3 | -0.3236 | 3/0/0 | +5.32 | 0/0/3 |
| iterative_ucb_excpq | alpha=1e-05 | 3 | -0.0006 | 3/0/0 | +0.17 | 2/0/1 |
| iterative_ucb_excpq | alpha=5e-05 | 3 | +0.0000 | 2/0/1 | +0.82 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=0.0002 | 3 | -0.0271 | 3/0/0 | +4.54 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=0.0004 | 3 | -0.0327 | 3/0/0 | +6.70 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=0.0006 | 3 | -0.0075 | 3/0/0 | +9.63 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=0.001 | 3 | -0.0410 | 3/0/0 | +11.80 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=0.01 | 3 | +0.0917 | 0/0/3 | +5.03 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=1e-05 | 3 | -0.0016 | 3/0/0 | +0.51 | 0/0/3 |
| probe_topk_m1_exact_global_excpq | alpha=5e-05 | 3 | -0.0057 | 3/0/0 | +1.56 | 0/0/3 |
