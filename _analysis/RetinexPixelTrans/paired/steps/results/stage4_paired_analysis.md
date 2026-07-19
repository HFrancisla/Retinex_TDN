# RetinexPixelTrans LOLv2 paired stage4 analysis

Scope: `_train/train_lolv2_stage4_paired.sh` 10 runs.

Important caveat: training logs reached `10000/10000`, but `img/10000` folders were deleted by cleanup after validation. Current on-disk decomposition artifacts are `img/2000` for most runs and `img/4000` for E5/E6, so this report ranks the currently saved visual artifacts, not final checkpoint outputs.

## Corrected Ranking

| rank | id | label | iter | Rlow->high PSNR | Rhigh->high PSNR | R/high | >high+0.1 | Rlow/Rhigh PSNR | Slo PSNR | Shi PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | E1 | smooth v1 0.6 | 2000 | 5.37 | 5.54 | 2.51 | 0.979 | 25.75 | 33.78 | 26.01 |
| 32 | E3 | smooth v1 0.8 | 2000 | 5.30 | 5.43 | 2.52 | 0.981 | 25.81 | 33.60 | 25.51 |
| 39 | E2 | smooth v1 0.7 | 2000 | 5.21 | 5.30 | 2.54 | 0.981 | 26.19 | 33.65 | 25.47 |
| 44 | E7 | equal_r 0.05 | 2000 | 5.21 | 5.24 | 2.53 | 0.979 | 25.58 | 33.96 | 25.60 |
| 56 | E4 | smooth v1 1.0 | 2000 | 4.88 | 4.86 | 2.60 | 0.983 | 27.11 | 33.27 | 24.81 |
| 65 | E10 | cross 0.005 | 2000 | 4.47 | 4.50 | 2.68 | 0.985 | 30.85 | 33.36 | 25.05 |
| 74 | E9 | cross 0.003 | 2000 | 4.43 | 4.45 | 2.69 | 0.986 | 31.21 | 33.33 | 25.01 |
| 83 | E8 | equal_r 0.15 | 2000 | 4.40 | 4.41 | 2.70 | 0.987 | 32.56 | 33.21 | 24.90 |
| 91 | E6 | smooth v3 0.5 | 4000 | 3.82 | 3.81 | 2.81 | 0.988 | 57.39 | 33.73 | 24.11 |
| 102 | E5 | smooth v2 0.5 | 4000 | 3.81 | 3.81 | 2.81 | 0.988 | 55.92 | 33.71 | 24.00 |

## Notes

- Primary preference is high-reference-aware R fidelity. R_low/R_high consistency is secondary because two jointly over-bright R maps can look very consistent.
- `E1 smooth v1 0.6` is the best current candidate by corrected score and R-to-high-reference PSNR.
- `E5 smooth v2 0.5` and `E6 smooth v3 0.5` have very high R consistency, but the worst R-to-reference PSNR and the strongest over-bright behavior.
- All runs reconstruct low images well (`S_low` about 33.2-34.0 dB), so reconstruction PSNR alone is not discriminative for Retinex decomposition quality.
