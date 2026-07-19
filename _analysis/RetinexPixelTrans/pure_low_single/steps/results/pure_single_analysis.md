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
| 42 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv2_20260717-200532 | r1-a2-sm0.1v2 | 41.67 | 10.96 | 0.889 | 0.0355 | 0.004 | 0 | 13.84 | 1.42 |  |
| 46 | LOLv2_1.0r_0.05anchorv1_0.05bdsp_0.0smv1_20260714-024244 | r1-a1-sm0v1 | 41.98 | 10.97 | 0.895 | 0.0229 | 0.004 | 12 | 13.86 | 1.41 |  |
| 58 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv3_20260718-094046 | r1-a2-sm0.1v3 | 41.67 | 10.98 | 0.891 | 0.0353 | 0.004 | 0 | 13.80 | 1.42 |  |
| 65 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.0smv1_20260714-060043 | r1-a2-sm0v1 | 41.93 | 11.00 | 0.895 | 0.0356 | 0.004 | 0 | 13.83 | 1.42 |  |
| 80 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv1_20260714-155441 | r1-a2-sm0.1v1 | 41.61 | 11.00 | 0.891 | 0.0350 | 0.005 | 0 | 13.76 | 1.43 |  |
| 91 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.5smv1_20260714-191342 | r1-a2-sm0.5v1 | 41.19 | 11.20 | 0.879 | 0.0341 | 0.007 | 0 | 13.58 | 1.44 |  |
| 116 | LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.1smv1_20260714-223238 | r0.3-a2-sm0.1v1 | 37.00 | 11.53 | 0.893 | 0.0176 | 0.021 | 0 | 12.03 | 1.64 |  |
| 117 | LOLv2_1.0r_0.05anchorv2_0.0bdsp_0.1smv3_20260718-130429 | r1-a2-sm0.1v3 | 36.88 | 2.61 | 0.925 | 0.0090 | 0.102 | 0 | 7.42 | 2.14 |  |
| 117 | LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.5smv1_20260715-015134 | r0.3-a2-sm0.5v1 | 36.60 | 11.70 | 0.874 | 0.0181 | 0.024 | 0 | 12.06 | 1.64 |  |

Dataset verdict: current conservative top is `LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv2_20260717-200532` (`r1-a2-sm0.1v2`).
