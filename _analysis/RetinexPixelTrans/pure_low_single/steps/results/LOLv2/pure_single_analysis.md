# Step 02 pure-low-single summary (LOLv2)

Image set: `auto`

Ranking is computed per dataset. LOLv2 high-reference metrics are diagnostic only; BDDnight is ranked without high-reference metrics.

`anchor_abs_error` is reported but not used as a cross-version rank term because anchor v1/v2 have different targets. Non-v2 anchor runs get a small canonical-anchor penalty so old ablations do not outrank the current v2 baseline solely on a different anchor definition.

## LOLv2 ranking

| score | run | label | self PSNR | R TV/input | corr(L,I) | anchor err | R>0.95 | anchor penalty | R→high PSNR | R/high | full recon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 119 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.02rtv_0.0rcons_0.0rsat_20260729-013835 | r1-a3-sm0.1v4-0.02rtv | 40.97 | 8.20 | 0.868 | 0.0517 | 0.002 | 12 | 17.02 | 1.17 |  |
| 133 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.03rtv_0.0rcons_0.0rsat_20260729-080740 | r1-a3-sm0.1v4-0.03rtv | 38.04 | 7.26 | 0.873 | 0.0458 | 0.001 | 12 | 17.39 | 1.13 |  |
| 135 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.0225rtv_0.0rcons_0.0rsat_20260730-041754 | r1-a3-sm0.1v4-0.0225rtv | 40.86 | 8.15 | 0.866 | 0.0532 | 0.002 | 12 | 17.07 | 1.17 |  |
| 151 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.05smv4_0.02rtv_0.0rcons_0.0rsat_20260729-112205 | r1-a3-sm0.05v4-0.02rtv | 41.02 | 8.25 | 0.875 | 0.0509 | 0.002 | 12 | 17.00 | 1.17 |  |
| 155 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.025rtv_0.0rcons_0.0rsat_20260729-045302 | r1-a3-sm0.1v4-0.025rtv | 37.84 | 7.70 | 0.874 | 0.0422 | 0.001 | 12 | 17.21 | 1.16 |  |
| 158 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.02rtv_0.0rcons_0.0rsat_20260727-112119 | r1-a3-sm0.1v4-0.02rtv | 40.94 | 8.32 | 0.868 | 0.0522 | 0.002 | 12 | 16.98 | 1.18 |  |
| 164 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.2smv4_0.02rtv_0.0rcons_0.0rsat_20260729-210529 | r1-a3-sm0.2v4-0.02rtv | 40.58 | 8.27 | 0.857 | 0.0519 | 0.002 | 12 | 17.08 | 1.18 |  |
| 171 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.02rtv_0.0rcons_0.0025rsat_20260730-104753 | r1-a3-sm0.1v4-0.02rtv-0.0025rsat | 40.87 | 8.25 | 0.868 | 0.0510 | 0.002 | 12 | 16.94 | 1.18 |  |
| 196 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.02rtv_0.01rcons_0.0rsat_20260730-073242 | r1-a3-sm0.1v4-0.02rtv-0.01rcons | 40.95 | 8.42 | 0.869 | 0.0492 | 0.002 | 12 | 16.78 | 1.19 |  |
| 199 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.0175rtv_0.0rcons_0.0rsat_20260730-010418 | r1-a3-sm0.1v4-0.0175rtv | 40.88 | 8.41 | 0.869 | 0.0497 | 0.002 | 12 | 16.86 | 1.19 |  |
| 201 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.075smv4_0.02rtv_0.0rcons_0.0rsat_20260729-143647 | r1-a3-sm0.075v4-0.02rtv | 37.57 | 7.96 | 0.879 | 0.0387 | 0.001 | 12 | 17.07 | 1.19 |  |
| 218 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.15smv4_0.02rtv_0.0rcons_0.0rsat_20260729-175106 | r1-a3-sm0.15v4-0.02rtv | 36.29 | 8.11 | 0.879 | 0.0352 | 0.000 | 12 | 17.13 | 1.20 |  |
| 229 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.0125rtv_0.0rcons_0.0rsat_20260728-191214 | r1-a3-sm0.1v4-0.0125rtv | 40.94 | 8.79 | 0.871 | 0.0474 | 0.002 | 12 | 16.60 | 1.22 |  |
| 235 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.015rtv_0.0rcons_0.0rsat_20260728-222456 | r1-a3-sm0.1v4-0.015rtv | 40.86 | 8.53 | 0.870 | 0.0487 | 0.002 | 12 | 16.81 | 1.20 |  |
| 255 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.01rtv_0.0rcons_0.0rsat_20260727-080939 | r1-a3-sm0.1v4-0.01rtv | 40.88 | 8.91 | 0.873 | 0.0460 | 0.003 | 12 | 16.43 | 1.23 |  |
| 270 | LOLv2_1.0r_0.05anchorv2_0.1bdsp_0.1smv2_20260724-080743 | r1-a2-sm0.1v2 | 41.28 | 10.09 | 0.883 | 0.0407 | 0.004 | 0 | 15.19 | 1.33 |  |
| 280 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.0rtv_0.0rcons_0.0rsat_20260727-045814 | r1-a3-sm0.1v4 | 40.88 | 9.90 | 0.874 | 0.0413 | 0.003 | 12 | 15.52 | 1.31 |  |
| 284 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.01rtv_0.05rcons_0.005rsat_20260727-174536 | r1-a3-sm0.1v4-0.01rtv-0.05rcons-0.005rsat | 41.20 | 10.02 | 0.875 | 0.0396 | 0.003 | 12 | 14.98 | 1.34 |  |
| 289 | LOLv2_1.0r_0.05anchorv2_0.08bdsp_0.1smv2_20260724-044405 | r1-a2-sm0.1v2 | 41.49 | 10.31 | 0.885 | 0.0390 | 0.004 | 0 | 14.78 | 1.36 |  |
| 295 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.01rtv_0.05rcons_0.0rsat_20260727-143329 | r1-a3-sm0.1v4-0.01rtv-0.05rcons | 40.12 | 9.62 | 0.874 | 0.0399 | 0.002 | 12 | 15.50 | 1.29 |  |
| 311 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv2_20260726-061302 | r1-a2-sm0.1v2 | 41.61 | 10.97 | 0.889 | 0.0354 | 0.004 | 0 | 13.81 | 1.42 |  |
| 320 | LOLv2_1.0r_0.05anchorv1_0.05bdsp_0.0smv1_20260723-151945 | r1-a1-sm0v1 | 41.98 | 10.98 | 0.895 | 0.0229 | 0.004 | 12 | 13.86 | 1.41 |  |
| 324 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv3_20260726-092924 | r1-a2-sm0.1v3 | 41.69 | 10.98 | 0.891 | 0.0353 | 0.005 | 0 | 13.80 | 1.42 |  |
| 334 | LOLv2_1.0r_0.05anchorv2_0.1bdsp_0.1smv2_0.0rtv_0.0rcons_0.0rsat_20260727-014711 | r1-a2-sm0.1v2 | 37.99 | 9.85 | 0.889 | 0.0306 | 0.003 | 0 | 15.30 | 1.36 |  |
| 337 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.0smv1_20260723-184140 | r1-a2-sm0v1 | 41.93 | 11.01 | 0.895 | 0.0354 | 0.004 | 0 | 13.79 | 1.42 |  |
| 353 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv1_20260723-215946 | r1-a2-sm0.1v1 | 41.60 | 11.02 | 0.891 | 0.0350 | 0.005 | 0 | 13.76 | 1.43 |  |
| 387 | LOLv2_1.0r_0.08anchorv2_0.08bdsp_0.1smv2_20260725-202313 | r1-a2-sm0.1v2 | 40.95 | 11.02 | 0.890 | 0.0317 | 0.007 | 0 | 13.51 | 1.46 |  |
| 416 | LOLv2_1.0r_0.08anchorv2_0.05bdsp_0.1smv2_20260724-145502 | r1-a2-sm0.1v2 | 41.04 | 11.71 | 0.893 | 0.0287 | 0.010 | 0 | 12.55 | 1.52 |  |
| 424 | LOLv2_1.0r_0.08anchorv2_0.05bdsp_0.05smv2_20260724-113129 | r1-a2-sm0.05v2 | 41.23 | 11.74 | 0.895 | 0.0289 | 0.010 | 0 | 12.56 | 1.53 |  |
| 441 | LOLv2_1.0r_0.1anchorv2_0.1bdsp_0.1smv2_20260726-025642 | r1-a2-sm0.1v2 | 40.14 | 11.20 | 0.893 | 0.0287 | 0.010 | 0 | 13.20 | 1.50 |  |
| 443 | LOLv2_1.0r_0.08anchorv2_0.05bdsp_0.2smv2_20260724-181848 | r1-a2-sm0.2v2 | 40.87 | 11.73 | 0.888 | 0.0286 | 0.012 | 0 | 12.54 | 1.53 |  |
| 506 | LOLv2_1.0r_0.1anchorv2_0.05bdsp_0.1smv2_20260725-234018 | r1-a2-sm0.1v2 | 40.66 | 12.06 | 0.896 | 0.0253 | 0.015 | 0 | 11.98 | 1.58 |  |
| 506 | LOLv2_0.7r_0.1anchorv2_0.08bdsp_0.1smv2_20260724-012046 | r0.7-a2-sm0.1v2 | 38.64 | 11.71 | 0.899 | 0.0215 | 0.018 | 0 | 12.27 | 1.59 |  |
| 509 | LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.1smv1_20260723-115707 | r0.3-a2-sm0.1v1 | 37.00 | 11.56 | 0.894 | 0.0176 | 0.021 | 0 | 12.00 | 1.64 |  |

Dataset verdict: current conservative top is `LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.02rtv_0.0rcons_0.0rsat_20260729-013835` (`r1-a3-sm0.1v4-0.02rtv`).
