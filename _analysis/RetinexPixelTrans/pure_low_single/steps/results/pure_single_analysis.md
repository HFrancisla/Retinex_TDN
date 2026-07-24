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
| 28 | LOLv2_1.0r_0.05anchorv2_0.1bdsp_0.1smv2_20260724-080743 | r1-a2-sm0.1v2 | 41.28 | 10.09 | 0.883 | 0.0407 | 0.004 | 0 | 15.19 | 1.33 |  |
| 38 | LOLv2_1.0r_0.05anchorv2_0.08bdsp_0.1smv2_20260724-044405 | r1-a2-sm0.1v2 | 41.49 | 10.31 | 0.885 | 0.0390 | 0.004 | 0 | 14.78 | 1.36 |  |
| 62 | LOLv2_1.0r_0.05anchorv1_0.05bdsp_0.0smv1_20260723-151945 | r1-a1-sm0v1 | 41.98 | 10.98 | 0.895 | 0.0229 | 0.004 | 12 | 13.86 | 1.41 |  |
| 66 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.0smv1_20260723-184140 | r1-a2-sm0v1 | 41.93 | 11.01 | 0.895 | 0.0354 | 0.004 | 0 | 13.79 | 1.42 |  |
| 72 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv1_20260723-215946 | r1-a2-sm0.1v1 | 41.60 | 11.02 | 0.891 | 0.0350 | 0.005 | 0 | 13.76 | 1.43 |  |
| 108 | LOLv2_1.0r_0.08anchorv2_0.05bdsp_0.1smv2_20260724-145502 | r1-a2-sm0.1v2 | 41.04 | 11.71 | 0.893 | 0.0287 | 0.010 | 0 | 12.55 | 1.52 |  |
| 115 | LOLv2_1.0r_0.08anchorv2_0.05bdsp_0.2smv2_20260724-181848 | r1-a2-sm0.2v2 | 40.87 | 11.73 | 0.888 | 0.0286 | 0.012 | 0 | 12.54 | 1.53 |  |
| 120 | LOLv2_1.0r_0.08anchorv2_0.05bdsp_0.05smv2_20260724-113129 | r1-a2-sm0.05v2 | 41.23 | 11.74 | 0.895 | 0.0289 | 0.010 | 0 | 12.56 | 1.53 |  |
| 138 | LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.1smv1_20260723-115707 | r0.3-a2-sm0.1v1 | 37.00 | 11.56 | 0.894 | 0.0176 | 0.021 | 0 | 12.00 | 1.64 |  |
| 145 | LOLv2_0.7r_0.1anchorv2_0.08bdsp_0.1smv2_20260724-012046 | r0.7-a2-sm0.1v2 | 38.64 | 11.71 | 0.899 | 0.0215 | 0.018 | 0 | 12.27 | 1.59 |  |

Dataset verdict: current conservative top is `LOLv2_1.0r_0.05anchorv2_0.1bdsp_0.1smv2_20260724-080743` (`r1-a2-sm0.1v2`).
