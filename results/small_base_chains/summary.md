# Pythia-410M base-reference compression-recovery chains

Regime R1, 10 cycles x R = 50 recovery steps (AdamW moments kept, constant lr 2e-06, batch 4 x 512). Values: held-out loss minus the no_compression chain of the same seed at the same step, 1e-3 nats, mean ± std over seeds. Uncompressed start loss 1.72117.

no_compression final loss per cycle (mean over seeds): 1.7185, 1.7177, 1.7156, 1.7121, 1.7124, 1.7098, 1.7074, 1.7042, 1.7008, 1.7010

## P = 0.95

### immediate (seeds per method: excp_style 3, inshrinkerator_style 3, dacp 3, first_order_ucb 3, iterative_ucb 3)

| cycle | excp_style | inshrinkerator_style | dacp | first_order_ucb | iterative_ucb |
|---|---|---|---|---|---|
| 1 | 362.2 ± 0.0 | 581.1 ± 8.0 | 318.3 ± 6.7 | 248.3 ± 3.1 | 110.9 ± 3.1 |
| 2 | 323.0 ± 0.0 | 475.5 ± 10.3 | 194.9 ± 2.5 | 147.2 ± 1.8 | 65.1 ± 2.5 |
| 3 | 286.7 ± 0.0 | 397.6 ± 13.0 | 138.7 ± 4.0 | 99.8 ± 1.4 | 42.9 ± 1.8 |
| 4 | 254.8 ± 0.0 | 347.9 ± 10.9 | 99.7 ± 4.6 | 68.6 ± 1.1 | 29.7 ± 1.2 |
| 5 | 227.8 ± 0.0 | 314.8 ± 5.2 | 79.2 ± 4.1 | 53.3 ± 1.1 | 24.2 ± 0.7 |
| 6 | 201.8 ± 0.0 | 286.3 ± 5.7 | 61.1 ± 4.6 | 38.5 ± 1.0 | 17.4 ± 0.5 |
| 7 | 179.9 ± 0.0 | 258.8 ± 8.9 | 51.9 ± 2.9 | 30.4 ± 0.9 | 14.8 ± 0.4 |
| 8 | 159.5 ± 0.0 | 252.6 ± 9.6 | 44.9 ± 3.2 | 24.8 ± 1.2 | 12.7 ± 0.5 |
| 9 | 143.2 ± 0.1 | 233.2 ± 9.2 | 40.0 ± 2.4 | 21.0 ± 1.8 | 12.3 ± 0.4 |
| 10 | 129.5 ± 0.1 | 228.0 ± 20.1 | 37.8 ± 3.3 | 19.1 ± 1.7 | 12.5 ± 0.6 |

Paired wins of iterative_ucb (immediate, seeds with lower loss / paired seeds):

| cycle | excp_style | inshrinkerator_style | dacp | first_order_ucb |
|---|---|---|---|---|
| 1 | 3/3 | 3/3 | 3/3 | 3/3 |
| 2 | 3/3 | 3/3 | 3/3 | 3/3 |
| 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 4 | 3/3 | 3/3 | 3/3 | 3/3 |
| 5 | 3/3 | 3/3 | 3/3 | 3/3 |
| 6 | 3/3 | 3/3 | 3/3 | 3/3 |
| 7 | 3/3 | 3/3 | 3/3 | 3/3 |
| 8 | 3/3 | 3/3 | 3/3 | 3/3 |
| 9 | 3/3 | 3/3 | 3/3 | 3/3 |
| 10 | 3/3 | 3/3 | 3/3 | 3/3 |

### final (seeds per method: excp_style 3, inshrinkerator_style 3, dacp 3, first_order_ucb 3, iterative_ucb 3)

| cycle | excp_style | inshrinkerator_style | dacp | first_order_ucb | iterative_ucb |
|---|---|---|---|---|---|
| 1 | 15.4 ± 0.1 | 64.2 ± 1.8 | 21.9 ± 1.8 | 11.4 ± 0.7 | 12.0 ± 0.1 |
| 2 | 12.1 ± 0.0 | 68.7 ± 1.4 | 20.4 ± 1.8 | 8.0 ± 0.8 | 8.5 ± 0.3 |
| 3 | 11.5 ± 0.3 | 69.3 ± 0.8 | 18.6 ± 2.5 | 6.9 ± 0.9 | 6.6 ± 0.5 |
| 4 | 15.9 ± 0.0 | 71.6 ± 1.4 | 21.3 ± 1.7 | 8.2 ± 1.1 | 7.4 ± 0.4 |
| 5 | 16.0 ± 0.1 | 75.3 ± 0.9 | 22.1 ± 2.3 | 8.2 ± 0.3 | 7.4 ± 0.5 |
| 6 | 12.4 ± 0.1 | 66.0 ± 2.0 | 18.4 ± 1.1 | 4.7 ± 0.9 | 4.5 ± 0.0 |
| 7 | 14.5 ± 0.1 | 69.4 ± 1.2 | 19.7 ± 1.1 | 6.4 ± 0.9 | 6.2 ± 0.2 |
| 8 | 17.5 ± 0.1 | 71.4 ± 1.1 | 21.9 ± 1.6 | 7.6 ± 1.7 | 7.5 ± 0.4 |
| 9 | 19.1 ± 0.3 | 75.1 ± 1.8 | 23.2 ± 1.1 | 9.7 ± 1.1 | 9.3 ± 0.5 |
| 10 | 18.7 ± 0.1 | 76.8 ± 1.0 | 25.1 ± 1.3 | 10.9 ± 1.1 | 10.8 ± 0.4 |

Paired wins of iterative_ucb (final, seeds with lower loss / paired seeds):

| cycle | excp_style | inshrinkerator_style | dacp | first_order_ucb |
|---|---|---|---|---|
| 1 | 3/3 | 3/3 | 3/3 | 0/3 |
| 2 | 3/3 | 3/3 | 3/3 | 0/3 |
| 3 | 3/3 | 3/3 | 3/3 | 2/3 |
| 4 | 3/3 | 3/3 | 3/3 | 2/3 |
| 5 | 3/3 | 3/3 | 3/3 | 3/3 |
| 6 | 3/3 | 3/3 | 3/3 | 2/3 |
| 7 | 3/3 | 3/3 | 3/3 | 2/3 |
| 8 | 3/3 | 3/3 | 3/3 | 2/3 |
| 9 | 3/3 | 3/3 | 3/3 | 2/3 |
| 10 | 3/3 | 3/3 | 3/3 | 2/3 |

Timing per cycle (mean over cycles and seeds):

| method | score s | cycle s | peak GiB |
|---|---|---|---|
| excp_style | 0.5 | 13.6 | 10.63 |
| inshrinkerator_style | 16.8 | 30.2 | 10.63 |
| dacp | 36.8 | 50.6 | 10.63 |
| first_order_ucb | 1.6 | 15.0 | 11.51 |
| iterative_ucb | 27.6 | 41.3 | 11.83 |

## P = 0.98

### immediate (seeds per method: excp_style 3, inshrinkerator_style 3, dacp 3, first_order_ucb 3, iterative_ucb 3)

| cycle | excp_style | inshrinkerator_style | dacp | first_order_ucb | iterative_ucb |
|---|---|---|---|---|---|
| 1 | 617.8 ± 0.0 | 703.0 ± 5.9 | 520.0 ± 7.7 | 462.2 ± 6.6 | 226.1 ± 4.2 |
| 2 | 582.3 ± 0.0 | 609.9 ± 7.3 | 384.3 ± 6.9 | 337.7 ± 6.4 | 149.9 ± 4.9 |
| 3 | 548.6 ± 0.0 | 531.7 ± 18.1 | 303.9 ± 4.6 | 264.4 ± 5.0 | 113.2 ± 4.1 |
| 4 | 519.0 ± 0.0 | 483.4 ± 11.4 | 248.9 ± 3.5 | 215.4 ± 3.2 | 90.1 ± 3.1 |
| 5 | 492.5 ± 0.1 | 445.3 ± 10.6 | 210.3 ± 5.0 | 178.7 ± 3.5 | 78.7 ± 2.8 |
| 6 | 465.6 ± 0.1 | 411.1 ± 7.1 | 174.0 ± 4.5 | 148.2 ± 2.8 | 67.3 ± 2.3 |
| 7 | 441.0 ± 0.1 | 384.4 ± 11.0 | 148.4 ± 3.5 | 125.5 ± 2.5 | 61.0 ± 2.0 |
| 8 | 416.4 ± 0.1 | 370.3 ± 18.0 | 130.5 ± 3.3 | 107.5 ± 2.2 | 56.4 ± 1.6 |
| 9 | 396.2 ± 0.1 | 348.7 ± 12.9 | 116.7 ± 2.7 | 96.1 ± 3.7 | 54.1 ± 1.7 |
| 10 | 377.2 ± 0.1 | 343.1 ± 17.4 | 107.1 ± 2.8 | 87.0 ± 3.2 | 52.4 ± 1.3 |

Paired wins of iterative_ucb (immediate, seeds with lower loss / paired seeds):

| cycle | excp_style | inshrinkerator_style | dacp | first_order_ucb |
|---|---|---|---|---|
| 1 | 3/3 | 3/3 | 3/3 | 3/3 |
| 2 | 3/3 | 3/3 | 3/3 | 3/3 |
| 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 4 | 3/3 | 3/3 | 3/3 | 3/3 |
| 5 | 3/3 | 3/3 | 3/3 | 3/3 |
| 6 | 3/3 | 3/3 | 3/3 | 3/3 |
| 7 | 3/3 | 3/3 | 3/3 | 3/3 |
| 8 | 3/3 | 3/3 | 3/3 | 3/3 |
| 9 | 3/3 | 3/3 | 3/3 | 3/3 |
| 10 | 3/3 | 3/3 | 3/3 | 3/3 |

### final (seeds per method: excp_style 3, inshrinkerator_style 3, dacp 3, first_order_ucb 3, iterative_ucb 3)

| cycle | excp_style | inshrinkerator_style | dacp | first_order_ucb | iterative_ucb |
|---|---|---|---|---|---|
| 1 | 48.9 ± 0.0 | 80.5 ± 1.9 | 51.4 ± 0.4 | 41.4 ± 1.3 | 44.9 ± 1.0 |
| 2 | 45.8 ± 0.1 | 78.9 ± 1.6 | 48.8 ± 1.9 | 38.2 ± 1.6 | 37.1 ± 0.4 |
| 3 | 41.9 ± 0.3 | 79.4 ± 2.4 | 44.2 ± 1.3 | 33.7 ± 0.5 | 32.9 ± 0.6 |
| 4 | 47.8 ± 0.2 | 81.4 ± 1.9 | 48.7 ± 1.9 | 37.5 ± 1.3 | 37.1 ± 0.5 |
| 5 | 51.0 ± 0.5 | 87.2 ± 1.4 | 50.2 ± 1.1 | 37.4 ± 1.7 | 36.9 ± 0.8 |
| 6 | 43.8 ± 0.4 | 78.5 ± 1.2 | 44.1 ± 1.2 | 32.1 ± 0.6 | 32.3 ± 0.8 |
| 7 | 46.7 ± 0.5 | 79.9 ± 0.6 | 45.9 ± 1.0 | 33.7 ± 1.2 | 32.9 ± 0.8 |
| 8 | 50.2 ± 0.2 | 82.2 ± 0.8 | 47.3 ± 0.1 | 35.6 ± 1.8 | 34.7 ± 0.5 |
| 9 | 51.4 ± 0.2 | 86.3 ± 1.5 | 49.5 ± 0.8 | 36.6 ± 1.2 | 35.9 ± 0.1 |
| 10 | 52.3 ± 0.1 | 87.2 ± 0.7 | 49.1 ± 0.8 | 38.3 ± 0.7 | 37.9 ± 0.8 |

Paired wins of iterative_ucb (final, seeds with lower loss / paired seeds):

| cycle | excp_style | inshrinkerator_style | dacp | first_order_ucb |
|---|---|---|---|---|
| 1 | 3/3 | 3/3 | 3/3 | 0/3 |
| 2 | 3/3 | 3/3 | 3/3 | 2/3 |
| 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 4 | 3/3 | 3/3 | 3/3 | 2/3 |
| 5 | 3/3 | 3/3 | 3/3 | 2/3 |
| 6 | 3/3 | 3/3 | 3/3 | 2/3 |
| 7 | 3/3 | 3/3 | 3/3 | 3/3 |
| 8 | 3/3 | 3/3 | 3/3 | 2/3 |
| 9 | 3/3 | 3/3 | 3/3 | 2/3 |
| 10 | 3/3 | 3/3 | 3/3 | 2/3 |

Timing per cycle (mean over cycles and seeds):

| method | score s | cycle s | peak GiB |
|---|---|---|---|
| excp_style | 0.5 | 13.9 | 10.63 |
| inshrinkerator_style | 17.2 | 30.9 | 10.63 |
| dacp | 36.0 | 49.4 | 10.63 |
| first_order_ucb | 1.6 | 14.5 | 11.51 |
| iterative_ucb | 27.3 | 40.3 | 11.83 |

