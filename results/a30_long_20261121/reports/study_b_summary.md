## BO

| seed | steps | method | mean deletion rate | after resumption: mean gap | lower than ExCP | checkpoints: mean gap | lower than ExCP | final gap | final PPL (no comp.) | restore damage at breaks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 7500 | ExCP | 0.816 | 37.3 |  | 27.0 |  | 27.2 | 32.30 (31.44) | 380/125/46/18/5 | 178.8 |
| 42 | 7500 | DACP-G_rho | 0.816 | 20.3 | 25/25 | 19.9 | 25/25 | 20.3 | 32.08 (31.44) | 29/-5/-9/-9/-4 | 7.3 |
| 42 | 7500 | DACP-G_rho 4x4 | 0.816 | 19.7 |  | 18.5 |  | 19.0 | 32.04 (31.44) | 44/12/1/-1/-1 | 21.6 |
| 42 | 7500 | Inshrinkerator | 0.816 | 209.9 |  | 128.7 |  | 170.2 | 37.27 (31.44) | 1059/450/331/260/205 | 554.3 |
| 43 | 2750 | ExCP | 0.801 | 58.4 |  | 29.8 |  | 39.6 | 44.36 (42.64) | 383/126 | 408.1 |
| 43 | 2750 | DACP-G_rho | 0.801 | 25.5 | 10/10 | 18.3 | 6/6 | 26.0 | 43.76 (42.64) | 36/-6 | 30.5 |
| 44 | 7500 | ExCP | 0.818 | 41.7 |  | 30.6 |  | 28.6 | 32.23 (31.32) | 397/140/73/22/8 | 168.8 |
| 44 | 7500 | DACP-G_rho | 0.818 | 25.1 | 25/25 | 23.4 | 25/25 | 21.2 | 32.00 (31.32) | 34/-7/-12/-8/-5 | 5.1 |
| 45 | 7500 | ExCP | 0.812 | 41.2 |  | 29.5 |  | 29.7 | 32.34 (31.40) | 435/176/87/35/8 | 187.6 |
| 45 | 7500 | DACP-G_rho | 0.812 | 18.1 | 25/25 | 17.1 | 25/25 | 17.2 | 31.94 (31.40) | 30/-3/-11/-8/-4 | 6.2 |

Across seeds run to step 7500 (mean ± std over seeds):

- ExCP (3 seeds: 42,44,45): after resumption 40.1 ± 2.4, checkpoints 29.0 ± 1.8, final 28.5 ± 1.3, restore damage over all checkpoints 178.4 ± 9.4
- DACP-G_rho (3 seeds: 42,44,45): after resumption 21.2 ± 3.6, checkpoints 20.1 ± 3.1, final 19.6 ± 2.1, restore damage over all checkpoints 6.2 ± 1.1
- DACP-G_rho 4x4 (1 seeds: 42): after resumption 19.7, checkpoints 18.5, final 19.0, restore damage over all checkpoints 21.6
- Inshrinkerator (1 seeds: 42): after resumption 209.9, checkpoints 128.7, final 170.2, restore damage over all checkpoints 554.3
- DACP-G_rho lower than ExCP: 75/75 points after resumption, 75/75 checkpoints

Across seeds run to step 2750 (mean ± std over seeds):

- ExCP (1 seeds: 43): after resumption 58.4, checkpoints 29.8, final 39.6, restore damage over all checkpoints 408.1
- DACP-G_rho (1 seeds: 43): after resumption 25.5, checkpoints 18.3, final 26.0, restore damage over all checkpoints 30.5
- DACP-G_rho lower than ExCP: 10/10 points after resumption, 6/6 checkpoints

## BC

| seed | steps | method | mean deletion rate | after resumption: mean gap | lower than ExCP | checkpoints: mean gap | lower than ExCP | final gap | final PPL (no comp.) | restore damage at breaks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 7500 | ExCP | 0.880 | 49.3 |  | 35.4 |  | 35.3 | 32.57 (31.44) | 502/270/173/101/43 | 310.9 |
| 42 | 7500 | DACP-G_rho | 0.880 | 41.3 | 25/25 | 32.8 | 25/25 | 32.2 | 32.47 (31.44) | 118/24/3/-1/1 | 79.7 |
| 42 | 7500 | DACP-G | 0.880 | 63.3 |  | 47.2 |  | 50.7 | 33.07 (31.44) | 136/51/19/7/4 | 54.1 |
| 42 | 2750 | noise | 0.001 | -2.4 |  | -2.1 |  | -1.5 | 42.67 (42.73) | -0/-0 | -0.1 |
| 44 | 7500 | ExCP | 0.880 | 51.8 |  | 36.9 |  | 35.3 | 32.45 (31.32) | 428/255/158/97/43 | 266.0 |
| 44 | 7500 | DACP-G_rho | 0.880 | 45.6 | 23/25 | 36.3 | 19/25 | 34.6 | 32.43 (31.32) | 131/23/1/-2/-1 | 65.0 |
| 45 | 7500 | ExCP | 0.880 | 49.5 |  | 34.6 |  | 34.4 | 32.50 (31.40) | 511/282/178/106/52 | 298.0 |
| 45 | 7500 | DACP-G_rho | 0.880 | 39.8 | 25/25 | 31.6 | 25/25 | 32.5 | 32.44 (31.40) | 125/25/2/-1/1 | 72.5 |

Across seeds run to step 7500 (mean ± std over seeds):

- ExCP (3 seeds: 42,44,45): after resumption 50.2 ± 1.4, checkpoints 35.6 ± 1.2, final 35.0 ± 0.5, restore damage over all checkpoints 291.6 ± 23.1
- DACP-G_rho (3 seeds: 42,44,45): after resumption 42.2 ± 3.0, checkpoints 33.6 ± 2.4, final 33.1 ± 1.3, restore damage over all checkpoints 72.4 ± 7.3
- DACP-G (1 seeds: 42): after resumption 63.3, checkpoints 47.2, final 50.7, restore damage over all checkpoints 54.1
- DACP-G_rho lower than ExCP: 73/75 points after resumption, 69/75 checkpoints

Across seeds run to step 2750 (mean ± std over seeds):

- noise (1 seeds: 42): after resumption -2.4, checkpoints -2.1, final -1.5, restore damage over all checkpoints -0.1

