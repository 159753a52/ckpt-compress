## Reference

- iter_R2_a_s42.json (commit 9f2d62f): checkpoint PPL 5.3484, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- iter_R2_a_s43.json (commit 9f2d62f): checkpoint PPL 5.3484, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- iter_R2_a_s44.json (commit 9f2d62f): checkpoint PPL 5.3484, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- iter_R2_b_s42.json (commit 9f2d62f): checkpoint PPL 5.3484, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- iter_R2_b_s43.json (commit 9f2d62f): checkpoint PPL 5.3484, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)
- iter_R2_b_s44.json (commit 9f2d62f): checkpoint PPL 5.3484, pretrained base PPL 14.4560, fp32 weights+exp_avg+exp_avg_sq 4864.0 MB (405334016 parameters)

## Bytes vs restored perplexity

| method | operating point | n | eligible sparsity | restored PPL | zlib MB | lzma MB | torch.save MB | CR (fp32/zlib) | w / m / v zlib MB |
|---|---|---|---|---|---|---|---|---|---|
| excp_pipeline | alpha=0.0002 | 3 | 0.2213 | 5.3331 +- 0.0003 | 247.64 | 248.18 | 608.5 | 19.6 | 173.3 / 37.7 / 36.6 |
| excp_pipeline | alpha=0.0004 | 3 | 0.4228 | 5.3208 +- 0.0007 | 215.19 | 215.82 | 608.5 | 22.6 | 155.3 / 30.4 / 29.5 |
| excp_pipeline | alpha=0.0006 | 3 | 0.5908 | 5.3141 +- 0.0014 | 178.95 | 177.69 | 608.6 | 27.2 | 132.8 / 23.4 / 22.7 |
| excp_pipeline | alpha=0.001 | 3 | 0.8101 | 5.5146 +- 0.0006 | 113.41 | 110.00 | 608.7 | 42.9 | 87.9 / 13.0 / 12.5 |
| excp_pipeline | alpha=0.01 | 3 | 0.9023 | 6.1230 +- 0.0005 | 79.47 | 75.40 | 610.0 | 61.2 | 62.7 / 8.5 / 8.2 |
| excp_pipeline | alpha=1e-05 | 3 | 0.0112 | 5.3347 +- 0.0011 | 273.70 | 271.87 | 608.5 | 17.8 | 186.3 / 44.4 / 43.0 |
| excp_pipeline | alpha=5e-05 | 3 | 0.0562 | 5.3358 +- 0.0003 | 271.25 | 269.61 | 608.5 | 17.9 | 186.4 / 43.0 / 41.8 |
| first_order_excpq | alpha=0.01,eligible_sp=0.95 (beyond ExCP cap) | 3 | 0.9500 | 7.7764 +- 0.0619 | 62.04 | 56.94 | 610.0 | 78.4 | 45.3 / 8.5 / 8.3 |
| first_order_excpq | alpha=0.01,eligible_sp=0.98 (beyond ExCP cap) | 3 | 0.9800 | 9.6285 +- 0.0598 | 43.38 | 39.33 | 610.0 | 112.1 | 32.7 / 5.4 / 5.2 |
| first_order_excpq | alpha=0.01,eligible_sp=0.99 (beyond ExCP cap) | 3 | 0.9900 | 10.6959 +- 0.0837 | 35.36 | 32.47 | 610.0 | 137.6 | 27.5 / 4.0 / 3.8 |
| iterative_first_order_excpq | alpha=0.0002 | 3 | 0.2213 | 5.3098 +- 0.0005 | 251.94 | 251.71 | 608.5 | 19.3 | 173.8 / 39.6 / 38.6 |
| iterative_first_order_excpq | alpha=0.0004 | 3 | 0.4228 | 5.2904 +- 0.0017 | 221.19 | 222.10 | 608.5 | 22.0 | 154.6 / 33.7 / 32.9 |
| iterative_first_order_excpq | alpha=0.0006 | 3 | 0.5908 | 5.3077 +- 0.0034 | 187.23 | 186.20 | 608.6 | 26.0 | 132.6 / 27.6 / 27.0 |
| iterative_first_order_excpq | alpha=0.001 | 3 | 0.8101 | 5.6794 +- 0.0122 | 123.83 | 120.13 | 608.7 | 39.3 | 88.7 / 17.7 / 17.4 |
| iterative_first_order_excpq | alpha=0.01 | 3 | 0.9023 | 6.4707 +- 0.0390 | 84.92 | 80.04 | 610.0 | 57.3 | 61.0 / 12.1 / 11.8 |
| iterative_first_order_excpq | alpha=0.01,eligible_sp=0.95 (beyond ExCP cap) | 3 | 0.9500 | 7.1737 +- 0.0777 | 61.96 | 56.81 | 610.0 | 78.5 | 45.4 / 8.4 / 8.2 |
| iterative_first_order_excpq | alpha=0.01,eligible_sp=0.98 (beyond ExCP cap) | 3 | 0.9800 | 7.8981 +- 0.0870 | 43.14 | 39.09 | 610.0 | 112.8 | 32.6 / 5.4 / 5.1 |
| iterative_first_order_excpq | alpha=0.01,eligible_sp=0.99 (beyond ExCP cap) | 3 | 0.9900 | 8.5340 +- 0.0585 | 35.09 | 32.24 | 610.0 | 138.6 | 27.3 / 4.0 / 3.8 |
| iterative_first_order_excpq | alpha=1e-05 | 3 | 0.0112 | 5.3341 +- 0.0008 | 273.77 | 271.91 | 608.5 | 17.8 | 186.1 / 44.6 / 43.1 |
| iterative_first_order_excpq | alpha=5e-05 | 3 | 0.0562 | 5.3312 +- 0.0008 | 272.74 | 270.91 | 608.5 | 17.8 | 186.9 / 43.6 / 42.3 |
| iterative_ucb_excpq | alpha=0.0002 | 3 | 0.2213 | 5.3322 +- 0.0002 | 251.07 | 250.61 | 608.5 | 19.4 | 172.1 / 40.0 / 38.9 |
| iterative_ucb_excpq | alpha=0.0004 | 3 | 0.4228 | 5.3229 +- 0.0003 | 220.76 | 221.38 | 608.5 | 22.0 | 152.9 / 34.4 / 33.5 |
| iterative_ucb_excpq | alpha=0.0006 | 3 | 0.5908 | 5.3162 +- 0.0024 | 187.12 | 185.91 | 608.6 | 26.0 | 130.8 / 28.5 / 27.8 |
| iterative_ucb_excpq | alpha=0.001 | 3 | 0.8101 | 5.4636 +- 0.0048 | 124.10 | 120.26 | 608.7 | 39.2 | 87.5 / 18.5 / 18.1 |
| iterative_ucb_excpq | alpha=0.01 | 3 | 0.9023 | 6.0737 +- 0.0188 | 85.39 | 80.31 | 610.0 | 57.0 | 60.1 / 12.8 / 12.5 |
| iterative_ucb_excpq | alpha=0.01,eligible_sp=0.95 (beyond ExCP cap) | 3 | 0.9500 | 6.7715 +- 0.0719 | 62.49 | 57.14 | 610.0 | 77.8 | 44.7 / 9.1 / 8.7 |
| iterative_ucb_excpq | alpha=0.01,eligible_sp=0.98 (beyond ExCP cap) | 3 | 0.9800 | 7.6745 +- 0.0701 | 43.59 | 39.35 | 610.0 | 111.6 | 32.1 / 5.9 / 5.6 |
| iterative_ucb_excpq | alpha=0.01,eligible_sp=0.99 (beyond ExCP cap) | 3 | 0.9900 | 8.4375 +- 0.0736 | 35.36 | 32.38 | 610.0 | 137.6 | 26.9 / 4.4 / 4.1 |
| iterative_ucb_excpq | alpha=1e-05 | 3 | 0.0112 | 5.3344 +- 0.0012 | 273.59 | 271.77 | 608.5 | 17.8 | 185.9 / 44.5 / 43.2 |
| iterative_ucb_excpq | alpha=5e-05 | 3 | 0.0562 | 5.3357 +- 0.0006 | 272.09 | 270.30 | 608.5 | 17.9 | 186.0 / 43.7 / 42.4 |

## Paired vs excp_pipeline (same alpha and seed; wins/ties/losses, lower is better)

| method | operating point | n | d PPL (mean) | PPL w/t/l | d zlib MB (mean) | bytes w/t/l |
|---|---|---|---|---|---|---|
| iterative_first_order_excpq | alpha=0.0002 | 3 | -0.0232 | 3/0/0 | +4.30 | 0/0/3 |
| iterative_first_order_excpq | alpha=0.0004 | 3 | -0.0304 | 3/0/0 | +6.00 | 0/0/3 |
| iterative_first_order_excpq | alpha=0.0006 | 3 | -0.0063 | 3/0/0 | +8.28 | 0/0/3 |
| iterative_first_order_excpq | alpha=0.001 | 3 | +0.1649 | 0/0/3 | +10.41 | 0/0/3 |
| iterative_first_order_excpq | alpha=0.01 | 3 | +0.3476 | 0/0/3 | +5.46 | 0/0/3 |
| iterative_first_order_excpq | alpha=1e-05 | 3 | -0.0005 | 3/0/0 | +0.07 | 1/0/2 |
| iterative_first_order_excpq | alpha=5e-05 | 3 | -0.0047 | 3/0/0 | +1.49 | 0/0/3 |
| iterative_ucb_excpq | alpha=0.0002 | 3 | -0.0008 | 3/0/0 | +3.44 | 0/0/3 |
| iterative_ucb_excpq | alpha=0.0004 | 3 | +0.0021 | 0/0/3 | +5.56 | 0/0/3 |
| iterative_ucb_excpq | alpha=0.0006 | 3 | +0.0022 | 0/0/3 | +8.17 | 0/0/3 |
| iterative_ucb_excpq | alpha=0.001 | 3 | -0.0509 | 3/0/0 | +10.68 | 0/0/3 |
| iterative_ucb_excpq | alpha=0.01 | 3 | -0.0493 | 3/0/0 | +5.92 | 0/0/3 |
| iterative_ucb_excpq | alpha=1e-05 | 3 | -0.0002 | 2/0/1 | -0.11 | 2/0/1 |
| iterative_ucb_excpq | alpha=5e-05 | 3 | -0.0001 | 2/0/1 | +0.83 | 0/0/3 |
