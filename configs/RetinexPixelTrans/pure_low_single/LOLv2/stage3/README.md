# Stage 3 LOLv2 Pure-Low-Single Local Search

This stage continues from the strongest Stage 2 basin:
`recon=1.0, anchor=0.05 v3, bdsp=0.1, smooth=0.1 v4, r_tv=0.02`.

LOLv2 high images are still used only for validation checkpoint selection, not
for the training loss. All configs select checkpoints by
`r_low_highref_psnr=max`.

## Experiment Matrix

| Exp | Purpose | Main loss change |
|---|---|---|
| A | R_TV lower-middle probe | `r_tv_weight=0.0125` |
| B | R_TV lower-middle probe | `r_tv_weight=0.015` |
| C | Current best retest | `r_tv_weight=0.02` |
| D | R_TV upper probe | `r_tv_weight=0.025` |
| E | R_TV upper stress | `r_tv_weight=0.03` |
| F | Smooth lower probe | `smooth_weight=0.05`, `r_tv_weight=0.02` |
| G | Smooth lower-middle probe | `smooth_weight=0.075`, `r_tv_weight=0.02` |
| H | Smooth upper probe | `smooth_weight=0.15`, `r_tv_weight=0.02` |
| I | Smooth upper stress | `smooth_weight=0.2`, `r_tv_weight=0.02` |

The `smooth=0.1, r_tv=0.02` point is included only once as Exp C. Seed-repeat
configs are intentionally not included here because automatic experiment names
do not include `training.seed`; add explicit non-auto names after the best local
loss setting is selected.
