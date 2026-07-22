# Step 02 corrected paired analysis

Image set: `best`

Primary rule: rank by how close `R_low` and `R_high` are to matched `I_high`, then use `R_low/R_high` consistency as a secondary criterion.

This avoids the known failure mode where `R_low≈R_high≈over-bright` receives a high consistency score.

## Corrected top runs

| rank_score | run | label | Rlow→high PSNR | Rhigh→high PSNR | R/high | >high+0.1 | Rlow/Rhigh PSNR | Lhigh mean | corr(Llow,I) | corr(Lhigh,I) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 49 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.6smv1_20260720-020824 | cr=0.001 er=0.1 sm=0.6v1 | 18.51 | 30.73 | 1.29 | 0.495 | 20.01 | 0.929 | 0.822 | 0.363 |
| 52 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.7smv1_20260720-035020 | cr=0.001 er=0.1 sm=0.7v1 | 18.50 | 31.63 | 1.29 | 0.492 | 19.61 | 0.948 | 0.810 | 0.262 |
| 64 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_1.0smv1_20260720-071136 | cr=0.001 er=0.1 sm=1.0v1 | 18.44 | 31.87 | 1.29 | 0.489 | 19.32 | 0.960 | 0.796 | 0.122 |
| 65 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.8smv1_20260720-053058 | cr=0.001 er=0.1 sm=0.8v1 | 18.50 | 31.60 | 1.29 | 0.495 | 19.59 | 0.951 | 0.806 | 0.185 |
| 67 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.05er_0.5smv1_20260720-121319 | cr=0.001 er=0.05 sm=0.5v1 | 18.44 | 30.57 | 1.28 | 0.490 | 19.78 | 0.929 | 0.827 | 0.409 |
| 77 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.5smv1_20260713-075255 | cr=0.001 er=0.1 sm=0.5v1 | 18.45 | 29.81 | 1.30 | 0.508 | 20.14 | 0.916 | 0.827 | 0.475 |
| 97 | LOLv2_1.0rh_0.3rl_0.005crh_0.005crl_0.1er_0.5smv1_20260720-171701 | cr=0.005 er=0.1 sm=0.5v1 | 18.33 | 27.73 | 1.30 | 0.518 | 20.59 | 0.887 | 0.827 | 0.576 |
| 108 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.2er_0.8smv1_20260722-153450 | cr=0.001 er=0.2 sm=0.8v1 | 18.26 | 29.88 | 1.32 | 0.536 | 19.91 | 0.927 | 0.807 | 0.312 |
| 111 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.2er_0.6smv1_20260722-102830 | cr=0.001 er=0.2 sm=0.6v1 | 18.16 | 28.47 | 1.32 | 0.533 | 20.21 | 0.904 | 0.811 | 0.560 |
| 112 | LOLv2_1.0rh_0.3rl_0.003crh_0.003crl_0.1er_0.5smv1_20260720-153620 | cr=0.003 er=0.1 sm=0.5v1 | 18.16 | 28.70 | 1.32 | 0.535 | 20.21 | 0.899 | 0.830 | 0.524 |

## Consistency-only top runs

| run | label | Rlow/Rhigh PSNR | Rlow→high PSNR | R/high | >high+0.1 |
| --- | --- | --- | --- | --- | --- |
| LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.3er_0.1smv3_20260717-160114 | cr=0.05 er=0.3 sm=0.1v3 | 26.07 | 6.25 | 2.37 | 0.978 |
| LOLv2_1.0rh_0.3rl_0.1crh_0.1crl_0.2er_0.1smv3_20260717-141642 | cr=0.1 er=0.2 sm=0.1v3 | 25.85 | 6.31 | 2.36 | 0.978 |
| LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.2er_0.1smv3_20260717-123101 | cr=0.05 er=0.2 sm=0.1v3 | 25.75 | 6.39 | 2.35 | 0.977 |
| LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.2er_0.1smv3_20260717-084100 | cr=0.01 er=0.2 sm=0.1v3 | 25.45 | 6.54 | 2.33 | 0.977 |
| LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.2er_0.1smv2_20260717-070107 | cr=0.01 er=0.2 sm=0.1v2 | 25.34 | 6.58 | 2.32 | 0.977 |
| LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.1er_0.1smv2_20260717-034129 | cr=0.05 er=0.1 sm=0.1v2 | 25.09 | 6.94 | 2.26 | 0.975 |
| LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.2er_0.2smv3_20260717-174512 | cr=0.05 er=0.2 sm=0.2v3 | 25.00 | 6.93 | 2.27 | 0.976 |
| LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.2er_0.1smv2_20260717-052115 | cr=0.001 er=0.2 sm=0.1v2 | 24.81 | 6.82 | 2.29 | 0.976 |
| LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.1er_0.1smv2_20260717-015927 | cr=0.01 er=0.1 sm=0.1v2 | 24.54 | 7.36 | 2.21 | 0.973 |
| LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.1smv2_20260712-163016 | cr=0.001 er=0.1 sm=0.1v2 | 24.17 | 7.74 | 2.16 | 0.971 |

## Main verdict

- Corrected best run: `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.6smv1_20260720-020824` (`cr=0.001 er=0.1 sm=0.6v1`).
- Consistency-only best run: `LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.3er_0.1smv3_20260717-160114` (`cr=0.05 er=0.3 sm=0.1v3`).
- The two best runs differ, so the old consistency-only ranking is not sufficient for this batch.
