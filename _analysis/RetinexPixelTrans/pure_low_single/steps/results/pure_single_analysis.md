# Step 02 pure-low-single summary

Image set: `auto`

Ranking is computed per dataset. LOLv2 high-reference metrics are diagnostic only; BDDnight is ranked without high-reference metrics.

`anchor_abs_error` is reported but not used as a cross-version rank term because anchor v1/v2 have different targets. Non-v2 anchor runs get a small canonical-anchor penalty so old ablations do not outrank the current v2 baseline solely on a different anchor definition.

## BDDnight ranking

| score | run | label | self PSNR | R TV/input | corr(L,I) | anchor err | R>0.95 | anchor penalty | R→high PSNR | R/high | full recon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 21 | BDDnight_1.0r_0.05anchorv2_0.05bdsp_0.0smv1_20260715-233953 | r1-a2-sm0v1 | 41.41 | 3.32 | 0.983 | 0.0440 | 0.035 | 0 |  |  | 0.00355 |
| 26 | BDDnight_1.0r_0.05anchorv2_0.05bdsp_0.1smv1_20260716-060543 | r1-a2-sm0.1v1 | 40.95 | 3.53 | 0.981 | 0.0464 | 0.026 | 0 |  |  | 0.00358 |
| 27 | BDDnight_1.0r_0.05anchorv2_0.05bdsp_0.5smv1_20260716-081309 | r1-a2-sm0.5v1 | 39.83 | 3.91 | 0.973 | 0.0488 | 0.021 | 0 |  |  | 0.00378 |
| 42 | BDDnight_0.3r_0.05anchorv2_0.05bdsp_0.1smv1_20260716-102043 | r0.3-a2-sm0.1v1 | 36.88 | 4.99 | 0.979 | 0.0298 | 0.104 | 0 |  |  | 0.00624 |
| 49 | BDDnight_0.3r_0.05anchorv2_0.05bdsp_0.5smv1_20260716-123043 | r0.3-a2-sm0.5v1 | 34.00 | 22.23 | 0.977 | 0.0258 | 0.122 | 0 |  |  | 0.00900 |

Dataset verdict: current conservative top is `BDDnight_1.0r_0.05anchorv2_0.05bdsp_0.0smv1_20260715-233953` (`r1-a2-sm0v1`).

## LOLv2 ranking

| score | run | label | self PSNR | R TV/input | corr(L,I) | anchor err | R>0.95 | anchor penalty | R→high PSNR | R/high | full recon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 89 | LOLv2_1.0r_0.05anchorv2_0.1bdsp_0.1smv2_20260721-095840 | r1-a2-sm0.1v2 | 40.99 | 10.06 | 0.884 | 0.0400 | 0.004 | 0 | 15.22 | 1.33 |  |
| 105 | LOLv2_1.0r_0.05anchorv2_0.08bdsp_0.1smv2_20260721-063944 | r1-a2-sm0.1v2 | 40.90 | 10.32 | 0.886 | 0.0378 | 0.004 | 0 | 14.80 | 1.36 |  |
| 122 | LOLv2_1.0r_0.08anchorv2_0.08bdsp_0.1smv2_20260721-131933 | r1-a2-sm0.1v2 | 31.17 | 7.29 | 0.714 | 0.0505 | 0.000 | 0 | 13.53 | 1.39 |  |
| 128 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv2_20260717-200532 | r1-a2-sm0.1v2 | 41.67 | 10.96 | 0.889 | 0.0355 | 0.004 | 0 | 13.84 | 1.42 |  |
| 132 | LOLv2_1.0r_0.05anchorv1_0.05bdsp_0.0smv1_20260714-024244 | r1-a1-sm0v1 | 41.98 | 10.97 | 0.895 | 0.0229 | 0.004 | 12 | 13.86 | 1.41 |  |
| 134 | LOLv2_1.0r_0.1anchorv2_0.1bdsp_0.1smv2_20260721-164101 | r1-a2-sm0.1v2 | 30.99 | 7.52 | 0.697 | 0.0487 | 0.000 | 0 | 13.46 | 1.40 |  |
| 137 | LOLv2_0.7r_0.1anchorv2_0.08bdsp_0.1smv2_20260722-024327 | r0.7-a2-sm0.1v2 | 30.80 | 7.79 | 0.675 | 0.0490 | 0.000 | 0 | 13.39 | 1.40 |  |
| 144 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv3_20260718-094046 | r1-a2-sm0.1v3 | 41.67 | 10.98 | 0.891 | 0.0353 | 0.004 | 0 | 13.80 | 1.42 |  |
| 151 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.0smv1_20260714-060043 | r1-a2-sm0v1 | 41.93 | 11.00 | 0.895 | 0.0356 | 0.004 | 0 | 13.83 | 1.42 |  |
| 154 | LOLv2_1.0r_0.08anchorv2_0.05bdsp_0.1smv2_20260720-202630 | r1-a2-sm0.1v2 | 31.31 | 6.32 | 0.721 | 0.0514 | 0.000 | 0 | 13.34 | 1.43 |  |
| 157 | LOLv2_1.0r_0.08anchorv2_0.05bdsp_0.2smv2_20260721-231952 | r1-a2-sm0.2v2 | 30.92 | 6.24 | 0.687 | 0.0415 | 0.000 | 0 | 13.37 | 1.43 |  |
| 166 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv1_20260714-155441 | r1-a2-sm0.1v1 | 41.61 | 11.00 | 0.891 | 0.0350 | 0.005 | 0 | 13.76 | 1.43 |  |
| 166 | LOLv2_1.0r_0.08anchorv2_0.05bdsp_0.05smv2_20260721-200034 | r1-a2-sm0.05v2 | 31.33 | 6.40 | 0.724 | 0.0520 | 0.000 | 0 | 13.32 | 1.43 |  |
| 170 | LOLv2_1.0r_0.1anchorv2_0.05bdsp_0.1smv2_20260720-235517 | r1-a2-sm0.1v2 | 31.23 | 6.12 | 0.713 | 0.0471 | 0.000 | 0 | 13.20 | 1.45 |  |
| 185 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.5smv1_20260714-191342 | r1-a2-sm0.5v1 | 41.19 | 11.20 | 0.879 | 0.0341 | 0.007 | 0 | 13.58 | 1.44 |  |
| 188 | LOLv2_1.0r_0.15anchorv2_0.05bdsp_0.1smv2_20260721-032103 | r1-a2-sm0.1v2 | 30.76 | 6.19 | 0.679 | 0.0355 | 0.000 | 0 | 12.96 | 1.48 |  |
| 223 | LOLv2_1.0r_0.05anchorv2_0.0bdsp_0.1smv3_20260718-130429 | r1-a2-sm0.1v3 | 36.88 | 2.61 | 0.925 | 0.0090 | 0.102 | 0 | 7.42 | 2.14 |  |
| 249 | LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.5smv1_20260715-015134 | r0.3-a2-sm0.5v1 | 36.60 | 11.70 | 0.874 | 0.0181 | 0.024 | 0 | 12.06 | 1.64 |  |
| 252 | LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.1smv1_20260714-223238 | r0.3-a2-sm0.1v1 | 37.00 | 11.53 | 0.893 | 0.0176 | 0.021 | 0 | 12.03 | 1.64 |  |

Dataset verdict: current conservative top is `LOLv2_1.0r_0.05anchorv2_0.1bdsp_0.1smv2_20260721-095840` (`r1-a2-sm0.1v2`).
