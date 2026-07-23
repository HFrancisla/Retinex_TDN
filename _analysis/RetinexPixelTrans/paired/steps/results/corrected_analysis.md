# Step 02 corrected paired analysis

Image set: `best`

Primary rule: rank by how close `R_low` and `R_high` are to matched `I_high`, then use `R_low/R_high` consistency as a secondary criterion.

This avoids the known failure mode where `R_low≈R_high≈over-bright` receives a high consistency score.

## Corrected top runs

| rank_score | run | label | Rlow→high PSNR | Rhigh→high PSNR | R/high | >high+0.1 | Rlow/Rhigh PSNR | Lhigh mean | corr(Llow,I) | corr(Lhigh,I) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 36 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.6smv1_20260720-020824 | cr=0.001 er=0.1 sm=0.6v1 | 18.51 | 30.73 | 1.29 | 0.495 | 20.01 | 0.929 | 0.822 | 0.363 |
| 39 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.7smv1_20260720-035020 | cr=0.001 er=0.1 sm=0.7v1 | 18.50 | 31.63 | 1.29 | 0.492 | 19.61 | 0.948 | 0.810 | 0.262 |
| 51 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_1.0smv1_20260720-071136 | cr=0.001 er=0.1 sm=1.0v1 | 18.44 | 31.87 | 1.29 | 0.489 | 19.32 | 0.960 | 0.796 | 0.122 |
| 52 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.8smv1_20260720-053058 | cr=0.001 er=0.1 sm=0.8v1 | 18.50 | 31.60 | 1.29 | 0.495 | 19.59 | 0.951 | 0.806 | 0.185 |
| 54 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.05er_0.5smv1_20260720-121319 | cr=0.001 er=0.05 sm=0.5v1 | 18.44 | 30.57 | 1.28 | 0.490 | 19.78 | 0.929 | 0.827 | 0.409 |
| 64 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.5smv1_20260713-075255 | cr=0.001 er=0.1 sm=0.5v1 | 18.45 | 29.81 | 1.30 | 0.508 | 20.14 | 0.916 | 0.827 | 0.475 |
| 84 | LOLv2_1.0rh_0.3rl_0.005crh_0.005crl_0.1er_0.5smv1_20260720-171701 | cr=0.005 er=0.1 sm=0.5v1 | 18.33 | 27.73 | 1.30 | 0.518 | 20.59 | 0.887 | 0.827 | 0.576 |
| 95 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.2er_0.8smv1_20260722-153450 | cr=0.001 er=0.2 sm=0.8v1 | 18.26 | 29.88 | 1.32 | 0.536 | 19.91 | 0.927 | 0.807 | 0.312 |
| 98 | LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.2er_0.6smv1_20260722-102830 | cr=0.001 er=0.2 sm=0.6v1 | 18.16 | 28.47 | 1.32 | 0.533 | 20.21 | 0.904 | 0.811 | 0.560 |
| 99 | LOLv2_1.0rh_0.3rl_0.003crh_0.003crl_0.1er_0.5smv1_20260720-153620 | cr=0.003 er=0.1 sm=0.5v1 | 18.16 | 28.70 | 1.32 | 0.535 | 20.21 | 0.899 | 0.830 | 0.524 |

## Consistency-only top runs

| run | label | Rlow/Rhigh PSNR | Rlow→high PSNR | R/high | >high+0.1 |
| --- | --- | --- | --- | --- | --- |
| LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.1smv1_20260722-070424 | cr=0.001 er=0.1 sm=0.1v1 | 23.95 | 8.01 | 2.13 | 0.971 |
| LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.3smv1_20260722-084625 | cr=0.001 er=0.1 sm=0.3v1 | 22.30 | 9.34 | 1.98 | 0.957 |
| LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.5er_0.6smv1_20260722-135247 | cr=0.001 er=0.5 sm=0.6v1 | 22.28 | 12.12 | 1.73 | 0.921 |
| LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.5er_0.8smv1_20260722-185856 | cr=0.001 er=0.5 sm=0.8v1 | 21.57 | 16.58 | 1.44 | 0.683 |
| LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.15er_0.5smv1_20260720-135547 | cr=0.001 er=0.15 sm=0.5v1 | 20.64 | 18.12 | 1.33 | 0.547 |
| LOLv2_1.0rh_0.3rl_0.005crh_0.005crl_0.1er_0.5smv1_20260720-171701 | cr=0.005 er=0.1 sm=0.5v1 | 20.59 | 18.33 | 1.30 | 0.518 |
| LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.3er_0.6smv1_20260722-121029 | cr=0.001 er=0.3 sm=0.6v1 | 20.46 | 17.74 | 1.36 | 0.594 |
| LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.2er_0.6smv1_20260722-102830 | cr=0.001 er=0.2 sm=0.6v1 | 20.21 | 18.16 | 1.32 | 0.533 |
| LOLv2_1.0rh_0.3rl_0.003crh_0.003crl_0.1er_0.5smv1_20260720-153620 | cr=0.003 er=0.1 sm=0.5v1 | 20.21 | 18.16 | 1.32 | 0.535 |
| LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.5smv1_20260713-075255 | cr=0.001 er=0.1 sm=0.5v1 | 20.14 | 18.45 | 1.30 | 0.508 |

## Main verdict

- Corrected best run: `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.6smv1_20260720-020824` (`cr=0.001 er=0.1 sm=0.6v1`).
- Consistency-only best run: `LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.1smv1_20260722-070424` (`cr=0.001 er=0.1 sm=0.1v1`).
- The two best runs differ, so the old consistency-only ranking is not sufficient for this batch.
