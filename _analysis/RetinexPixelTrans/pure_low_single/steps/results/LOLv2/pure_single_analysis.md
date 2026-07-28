# Step 02 pure-low-single summary (LOLv2)

Image set: `auto`

Ranking is computed per dataset. LOLv2 high-reference metrics are diagnostic only; BDDnight is ranked without high-reference metrics.

`anchor_abs_error` is reported but not used as a cross-version rank term because anchor v1/v2 have different targets. Non-v2 anchor runs get a small canonical-anchor penalty so old ablations do not outrank the current v2 baseline solely on a different anchor definition.

## LOLv2 ranking

| score | run | label | self PSNR | R TV/input | corr(L,I) | anchor err | R>0.95 | anchor penalty | R→high PSNR | R/high | full recon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 61 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.02rtv_0.0rcons_0.0rsat_20260727-112119 | r1-a3-sm0.1v4-0.02rtv | 40.94 | 8.32 | 0.868 | 0.0522 | 0.002 | 12 | 16.98 | 1.18 |  |
| 81 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.01rtv_0.0rcons_0.0rsat_20260727-080939 | r1-a3-sm0.1v4-0.01rtv | 40.88 | 8.91 | 0.873 | 0.0460 | 0.003 | 12 | 16.43 | 1.23 |  |
| 101 | LOLv2_1.0r_0.05anchorv2_0.1bdsp_0.1smv2_20260724-080743 | r1-a2-sm0.1v2 | 41.28 | 10.09 | 0.883 | 0.0407 | 0.004 | 0 | 15.19 | 1.33 |  |
| 105 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.0rtv_0.0rcons_0.0rsat_20260727-045814 | r1-a3-sm0.1v4 | 40.88 | 9.90 | 0.874 | 0.0413 | 0.003 | 12 | 15.52 | 1.31 |  |
| 107 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.01rtv_0.05rcons_0.0rsat_20260727-143329 | r1-a3-sm0.1v4-0.01rtv-0.05rcons | 40.12 | 9.62 | 0.874 | 0.0399 | 0.002 | 12 | 15.50 | 1.29 |  |
| 119 | LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.01rtv_0.05rcons_0.005rsat_20260727-174536 | r1-a3-sm0.1v4-0.01rtv-0.05rcons-0.005rsat | 41.20 | 10.02 | 0.875 | 0.0396 | 0.003 | 12 | 14.98 | 1.34 |  |
| 120 | LOLv2_1.0r_0.05anchorv2_0.08bdsp_0.1smv2_20260724-044405 | r1-a2-sm0.1v2 | 41.49 | 10.31 | 0.885 | 0.0390 | 0.004 | 0 | 14.78 | 1.36 |  |
| 135 | LOLv2_1.0r_0.05anchorv2_0.1bdsp_0.1smv2_0.0rtv_0.0rcons_0.0rsat_20260727-014711 | r1-a2-sm0.1v2 | 37.99 | 9.85 | 0.889 | 0.0306 | 0.003 | 0 | 15.30 | 1.36 |  |
| 142 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv2_20260726-061302 | r1-a2-sm0.1v2 | 41.61 | 10.97 | 0.889 | 0.0354 | 0.004 | 0 | 13.81 | 1.42 |  |
| 151 | LOLv2_1.0r_0.05anchorv1_0.05bdsp_0.0smv1_20260723-151945 | r1-a1-sm0v1 | 41.98 | 10.98 | 0.895 | 0.0229 | 0.004 | 12 | 13.86 | 1.41 |  |
| 155 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv3_20260726-092924 | r1-a2-sm0.1v3 | 41.69 | 10.98 | 0.891 | 0.0353 | 0.005 | 0 | 13.80 | 1.42 |  |
| 168 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.0smv1_20260723-184140 | r1-a2-sm0v1 | 41.93 | 11.01 | 0.895 | 0.0354 | 0.004 | 0 | 13.79 | 1.42 |  |
| 184 | LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv1_20260723-215946 | r1-a2-sm0.1v1 | 41.60 | 11.02 | 0.891 | 0.0350 | 0.005 | 0 | 13.76 | 1.43 |  |
| 209 | LOLv2_1.0r_0.08anchorv2_0.08bdsp_0.1smv2_20260725-202313 | r1-a2-sm0.1v2 | 40.95 | 11.02 | 0.890 | 0.0317 | 0.007 | 0 | 13.51 | 1.46 |  |
| 245 | LOLv2_1.0r_0.1anchorv2_0.1bdsp_0.1smv2_20260726-025642 | r1-a2-sm0.1v2 | 40.14 | 11.20 | 0.893 | 0.0287 | 0.010 | 0 | 13.20 | 1.50 |  |
| 247 | LOLv2_1.0r_0.08anchorv2_0.05bdsp_0.1smv2_20260724-145502 | r1-a2-sm0.1v2 | 41.04 | 11.71 | 0.893 | 0.0287 | 0.010 | 0 | 12.55 | 1.52 |  |
| 255 | LOLv2_1.0r_0.08anchorv2_0.05bdsp_0.05smv2_20260724-113129 | r1-a2-sm0.05v2 | 41.23 | 11.74 | 0.895 | 0.0289 | 0.010 | 0 | 12.56 | 1.53 |  |
| 256 | LOLv2_1.0r_0.08anchorv2_0.05bdsp_0.2smv2_20260724-181848 | r1-a2-sm0.2v2 | 40.87 | 11.73 | 0.888 | 0.0286 | 0.012 | 0 | 12.54 | 1.53 |  |
| 304 | LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.1smv1_20260723-115707 | r0.3-a2-sm0.1v1 | 37.00 | 11.56 | 0.894 | 0.0176 | 0.021 | 0 | 12.00 | 1.64 |  |
| 310 | LOLv2_0.7r_0.1anchorv2_0.08bdsp_0.1smv2_20260724-012046 | r0.7-a2-sm0.1v2 | 38.64 | 11.71 | 0.899 | 0.0215 | 0.018 | 0 | 12.27 | 1.59 |  |
| 313 | LOLv2_1.0r_0.1anchorv2_0.05bdsp_0.1smv2_20260725-234018 | r1-a2-sm0.1v2 | 40.66 | 12.06 | 0.896 | 0.0253 | 0.015 | 0 | 11.98 | 1.58 |  |

Dataset verdict: current conservative top is `LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.02rtv_0.0rcons_0.0rsat_20260727-112119` (`r1-a3-sm0.1v4-0.02rtv`).
