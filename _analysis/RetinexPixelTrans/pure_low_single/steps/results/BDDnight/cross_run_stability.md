# Step 03 cross-run stability (BDDnight)

Baseline per dataset: `recon=1.0, anchor=v2, smooth=0`.

## BDDnight

| run | label | R L1 | R PSNR | L L1 | ΔR mean | ΔR TV/input | Δcorr(L,I) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BDDnight_0.3r_0.05anchorv2_0.05bdsp_0.1smv1_20260716-102043 | r0.3-a2-sm0.1v1 | 0.0715 | 21.99 | 0.0149 | 0.0589 | 1.67 | -0.004 |
| BDDnight_0.3r_0.05anchorv2_0.05bdsp_0.5smv1_20260716-123043 | r0.3-a2-sm0.5v1 | 0.1092 | 17.94 | 0.0192 | 0.0876 | 18.92 | -0.007 |
| BDDnight_1.0r_0.05anchorv2_0.05bdsp_0.0smv1_20260715-233953 | r1-a2-sm0v1 | 0.0000 | 100.00 | 0.0000 | 0.0000 | 0.00 | 0.000 |
| BDDnight_1.0r_0.05anchorv2_0.05bdsp_0.1smv1_20260716-060543 | r1-a2-sm0.1v1 | 0.0116 | 36.25 | 0.0029 | -0.0103 | 0.22 | -0.002 |
| BDDnight_1.0r_0.05anchorv2_0.05bdsp_0.5smv1_20260716-081309 | r1-a2-sm0.5v1 | 0.0272 | 28.68 | 0.0082 | -0.0144 | 0.59 | -0.010 |
