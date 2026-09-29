# 32B repeated compression-recovery chains, reference = base weights

Start: saved step-200 checkpoint (theta, AdamW moments), DeepSeek-R1-Distill-Qwen-32B. Each cycle compresses theta - base at residual sparsity P, restores pruned coordinates to base, keeps the moments, evaluates, trains R = 50 steps on the next recovery chunks (same order for all methods) and evaluates again. Deltas in 1e-3 nats. `vs NC` = minus the no_compression chain at the same step; `vs 200` = minus the uncompressed step-200 checkpoint (val 1.73824, gate 2.05814).

## No-compression reference

| step | val | gate |
|---|---|---|
| 200 | 1.73824 | 2.05814 |
| 250 | 1.72998 | 2.09243 |
| 300 | 1.72302 | 2.07904 |
| 350 | 1.71696 | 2.06493 |

## P = 0.95

Validation loss delta vs NC (immediate / after recovery):

| method | c1 imm | c1 fin | c2 imm | c2 fin | c3 imm | c3 fin |
|---|---|---|---|---|---|---|
| ExCP-style (abs d) | +42.5 | -12.2 | +26.1 | -14.8 | +14.4 | -4.0 |
| Inshrinkerator-style (abs g*theta) | +81.6 | -7.6 | +61.8 | -12.7 | +52.8 | -1.0 |
| DACP (block HVP + Weibull) | +14.8 | -16.0 | -3.1 | -18.4 | -11.2 | -7.2 |
| first-order UCB one-shot (B=16) | +0.8 | -17.7 | -15.4 | -20.5 | -22.5 | -12.6 |
| ours: iterative UCB (K=4, B=4) | -1.8 | -18.2 | -16.4 | -20.7 | -23.2 | -10.9 |

Validation loss delta vs uncompressed step 200:

| method | c1 imm | c1 fin | c2 imm | c2 fin | c3 imm | c3 fin |
|---|---|---|---|---|---|---|
| no compression | +0.0 | -8.3 | -8.3 | -15.2 | -15.2 | -21.3 |
| ExCP-style (abs d) | +42.5 | -20.5 | +17.9 | -30.0 | -0.8 | -25.2 |
| Inshrinkerator-style (abs g*theta) | +81.6 | -15.8 | +53.5 | -27.9 | +37.6 | -22.3 |
| DACP (block HVP + Weibull) | +14.8 | -24.3 | -11.4 | -33.7 | -26.4 | -28.5 |
| first-order UCB one-shot (B=16) | +0.8 | -25.9 | -23.7 | -35.7 | -37.8 | -33.9 |
| ours: iterative UCB (K=4, B=4) | -1.8 | -26.5 | -24.6 | -35.9 | -38.4 | -32.2 |

Gate loss delta vs NC (immediate / after recovery):

| method | c1 imm | c1 fin | c2 imm | c2 fin | c3 imm | c3 fin |
|---|---|---|---|---|---|---|
| ExCP-style (abs d) | +54.1 | -22.2 | -4.5 | -21.7 | -12.6 | -39.3 |
| Inshrinkerator-style (abs g*theta) | +84.5 | -16.6 | +27.1 | -20.4 | +17.2 | -38.6 |
| DACP (block HVP + Weibull) | +19.8 | -26.6 | -34.9 | -25.6 | -43.2 | -43.6 |
| first-order UCB one-shot (B=16) | -1.0 | -27.4 | -49.4 | -29.7 | -56.5 | -47.7 |
| ours: iterative UCB (K=4, B=4) | -3.1 | -26.1 | -50.1 | -28.9 | -55.3 | -45.3 |

Per-cycle cost and residual (cycle 1 / 2 / 3):

| method | zero-residual frac. | residual L2 | scoring+selection s | stage s (ours) | score peak GiB (max dev) | recovery peak GiB | chosen payload GB |
|---|---|---|---|---|---|---|---|
| ExCP-style (abs d) | 0.162 / 0.359 / 0.358 | 41.97 / 32.38 / 33.52 | 24 / 32 / 31 | - | 69.4 / 68.9 / 68.9 | 85.7 / 85.7 / 85.7 | 6.58 / 6.59 / 6.50 |
| Inshrinkerator-style (abs g*theta) | 0.162 / 0.370 / 0.373 | 41.97 / 23.67 / 23.13 | 20 / 25 / 31 | - | 69.4 / 68.9 / 68.9 | 85.7 / 85.7 / 85.7 | 6.43 / 6.37 / 6.33 |
| DACP (block HVP + Weibull) | 0.162 / 0.367 / 0.365 | 41.97 / 27.03 / 27.12 | 285 / 358 / 317 | - | 88.8 / 88.3 / 88.3 | 85.7 / 85.7 / 85.7 | 6.35 / 6.58 / 6.55 |
| first-order UCB one-shot (B=16) | 0.162 / 0.365 / 0.364 | 41.97 / 28.52 / 29.15 | 1555 / 1567 / 1550 | - | 103.8 / 103.3 / 103.3 | 85.7 / 85.7 / 85.7 | 6.40 / 6.30 / 6.22 |
| ours: iterative UCB (K=4, B=4) | 0.162 / 0.365 / 0.365 | 41.97 / 28.28 / 28.87 | 2205 / 2074 / 2044 | 536,584,535,534 / 417,534,554,543 / 402,501,554,570 | 103.8 / 103.4 / 103.3 | 85.7 / 85.7 / 85.7 | 6.43 / 6.36 / 6.32 |

## P = 0.98

Validation loss delta vs NC (immediate / after recovery):

| method | c1 imm | c1 fin | c2 imm | c2 fin | c3 imm | c3 fin |
|---|---|---|---|---|---|---|
| ExCP-style (abs d) | +109.7 | -5.0 | +89.4 | -8.5 | +71.2 | +2.4 |
| Inshrinkerator-style (abs g*theta) | +120.4 | -4.4 | +103.8 | -9.2 | +94.7 | +1.5 |
| DACP (block HVP + Weibull) | +60.3 | -12.8 | +28.5 | -15.9 | +12.9 | -6.6 |

Validation loss delta vs uncompressed step 200:

| method | c1 imm | c1 fin | c2 imm | c2 fin | c3 imm | c3 fin |
|---|---|---|---|---|---|---|
| no compression | +0.0 | -8.3 | -8.3 | -15.2 | -15.2 | -21.3 |
| ExCP-style (abs d) | +109.7 | -13.2 | +81.2 | -23.7 | +55.9 | -18.9 |
| Inshrinkerator-style (abs g*theta) | +120.4 | -12.7 | +95.6 | -24.4 | +79.5 | -19.8 |
| DACP (block HVP + Weibull) | +60.3 | -21.1 | +20.3 | -31.1 | -2.3 | -27.9 |

Gate loss delta vs NC (immediate / after recovery):

| method | c1 imm | c1 fin | c2 imm | c2 fin | c3 imm | c3 fin |
|---|---|---|---|---|---|---|
| ExCP-style (abs d) | +130.4 | -13.5 | +65.4 | -16.2 | +51.5 | -34.4 |
| Inshrinkerator-style (abs g*theta) | +126.1 | -17.4 | +66.9 | -18.1 | +59.0 | -35.2 |
| DACP (block HVP + Weibull) | +65.6 | -23.4 | -7.0 | -25.3 | -23.2 | -44.8 |

Per-cycle cost and residual (cycle 1 / 2 / 3):

| method | zero-residual frac. | residual L2 | scoring+selection s | stage s (ours) | score peak GiB (max dev) | recovery peak GiB | chosen payload GB |
|---|---|---|---|---|---|---|---|
| ExCP-style (abs d) | 0.162 / 0.376 / 0.376 | 41.97 / 28.25 / 28.84 | 29 / 35 / 32 | - | 69.4 / 68.9 / 68.9 | 85.7 / 85.7 / 85.7 | 3.72 / 3.65 / 3.58 |
| Inshrinkerator-style (abs g*theta) | 0.162 / 0.380 / 0.383 | 41.97 / 21.69 / 21.34 | 30 / 27 / 29 | - | 69.4 / 68.9 / 68.9 | 85.7 / 85.7 / 85.7 | 3.13 / 3.13 / 3.28 |
| DACP (block HVP + Weibull) | 0.162 / 0.380 / 0.379 | 41.97 / 23.38 / 23.34 | 334 / 353 / 338 | - | 88.8 / 88.3 / 88.3 | 85.7 / 85.7 / 85.7 | 2.76 / 3.36 / 3.41 |

Train loss: per-step values are in each JSON (`cycles[].train_log`). Mean over the last 10 recovery steps of each cycle:

| chain | c1 | c2 | c3 |
|---|---|---|---|
| no compression | 1.6156 | 1.7137 | 1.8437 |
| ExCP-style (abs d) P=0.95 | 1.6129 | 1.7196 | 1.8362 |
| Inshrinkerator-style (abs g*theta) P=0.95 | 1.6128 | 1.7216 | 1.8463 |
| DACP (block HVP + Weibull) P=0.95 | 1.6061 | 1.7131 | 1.8339 |
| first-order UCB one-shot (B=16) P=0.95 | 1.6043 | 1.7092 | 1.8284 |
| ours: iterative UCB (K=4, B=4) P=0.95 | 1.6037 | 1.7107 | 1.8288 |
| ExCP-style (abs d) P=0.98 | 1.6188 | 1.7266 | 1.8471 |
| Inshrinkerator-style (abs g*theta) P=0.98 | 1.6194 | 1.7237 | 1.8541 |
| DACP (block HVP + Weibull) P=0.98 | 1.6126 | 1.7170 | 1.8422 |

Lines between evaluation points in the figure are linear interpolations; only the marked points are measured.
