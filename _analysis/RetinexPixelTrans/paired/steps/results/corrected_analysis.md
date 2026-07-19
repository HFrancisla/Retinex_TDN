# Step 02 corrected paired analysis

Image set: `best`

Primary rule: rank by how close `R_low` and `R_high` are to matched `I_high`, then use `R_low/R_high` consistency as a secondary criterion.

This avoids the known failure mode where `R_low≈R_high≈over-bright` receives a high consistency score.

## Corrected top runs

| rank_score | run | label | Rlow→high PSNR | Rhigh→high PSNR | R/high | >high+0.1 | Rlow/Rhigh PSNR | Lhigh mean | corr(Llow,I) | corr(Lhigh,I) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.5smv1_20260713-075255 | cr=0.001 er=0.1 sm=0.5v1 | 18.45 | 29.81 | 1.30 | 0.508 | 20.14 | 0.916 | 0.827 | 0.475 |
| 35 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.3smv1_20260713-061113 | cr=0.001 er=0.1 sm=0.3v1 | 9.24 | 9.92 | 1.99 | 0.960 | 22.24 | 0.525 | 0.893 | 0.917 |
| 40 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.3smv2_20260712-213703 | cr=0.001 er=0.1 sm=0.3v2 | 9.03 | 9.69 | 2.02 | 0.959 | 22.73 | 0.518 | 0.896 | 0.929 |
| 54 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.1smv2_20260712-163016 | cr=0.001 er=0.1 sm=0.1v2 | 7.74 | 8.18 | 2.16 | 0.971 | 24.17 | 0.482 | 0.931 | 0.954 |
| 57 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.1smv1_20260713-042929 | cr=0.001 er=0.1 sm=0.1v1 | 7.74 | 8.14 | 2.16 | 0.972 | 24.17 | 0.481 | 0.931 | 0.951 |
| 69 | LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.1er_0.1smv2_20260717-015927 | cr=0.01 er=0.1 sm=0.1v2 | 7.36 | 7.70 | 2.21 | 0.973 | 24.54 | 0.470 | 0.932 | 0.955 |
| 79 | LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.1er_0.1smv2_20260717-034129 | cr=0.05 er=0.1 sm=0.1v2 | 6.94 | 7.16 | 2.26 | 0.975 | 25.09 | 0.455 | 0.932 | 0.956 |
| 86 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.2er_0.1smv2_20260717-052115 | cr=0.001 er=0.2 sm=0.1v2 | 6.82 | 7.21 | 2.29 | 0.976 | 24.81 | 0.456 | 0.929 | 0.952 |
| 96 | LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.2er_0.1smv2_20260717-070107 | cr=0.01 er=0.2 sm=0.1v2 | 6.58 | 6.87 | 2.32 | 0.977 | 25.34 | 0.447 | 0.928 | 0.951 |
| 105 | LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.2er_0.1smv3_20260717-084100 | cr=0.01 er=0.2 sm=0.1v3 | 6.54 | 6.84 | 2.33 | 0.977 | 25.45 | 0.447 | 0.928 | 0.951 |

## Consistency-only top runs

| run | label | Rlow/Rhigh PSNR | Rlow→high PSNR | R/high | >high+0.1 |
| --- | --- | --- | --- | --- | --- |
| LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.3er_0.1smv3_20260717-160114 | cr=0.05 er=0.3 sm=0.1v3 | 60.55 | 3.82 | 2.81 | 0.988 |
| LOLv2_1.0rh_0.3rl_0.1crh_0.1crl_0.2er_0.1smv3_20260717-141642 | cr=0.1 er=0.2 sm=0.1v3 | 59.12 | 3.82 | 2.81 | 0.988 |
| LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.2er_0.2smv3_20260717-174512 | cr=0.05 er=0.2 sm=0.2v3 | 56.88 | 3.80 | 2.81 | 0.988 |
| LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.2er_0.1smv3_20260717-123101 | cr=0.05 er=0.2 sm=0.1v3 | 55.28 | 3.80 | 2.81 | 0.988 |
| LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.2er_0.1smv3_20260717-084100 | cr=0.01 er=0.2 sm=0.1v3 | 25.45 | 6.54 | 2.33 | 0.977 |
| LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.2er_0.1smv2_20260717-070107 | cr=0.01 er=0.2 sm=0.1v2 | 25.34 | 6.58 | 2.32 | 0.977 |
| LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.1er_0.1smv2_20260717-034129 | cr=0.05 er=0.1 sm=0.1v2 | 25.09 | 6.94 | 2.26 | 0.975 |
| LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.2er_0.1smv2_20260717-052115 | cr=0.001 er=0.2 sm=0.1v2 | 24.81 | 6.82 | 2.29 | 0.976 |
| LOLv2_1.0rh_0.3rl_0.01crh_0.01crl_0.1er_0.1smv2_20260717-015927 | cr=0.01 er=0.1 sm=0.1v2 | 24.54 | 7.36 | 2.21 | 0.973 |
| LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.1smv2_20260712-163016 | cr=0.001 er=0.1 sm=0.1v2 | 24.17 | 7.74 | 2.16 | 0.971 |

## Main verdict

- Corrected best run: `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.5smv1_20260713-075255` (`cr=0.001 er=0.1 sm=0.5v1`).
- Consistency-only best run: `LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.3er_0.1smv3_20260717-160114` (`cr=0.05 er=0.3 sm=0.1v3`).
- The two best runs differ, so the old consistency-only ranking is not sufficient for this batch.

## Missing details

- LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.6smv1_20260720-020824
