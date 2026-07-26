# Step 02 pure-low-single summary (BDDnight)

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
