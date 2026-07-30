# Stage 4 LOLv2 Pure-Low-Single Follow-Up

Stage 4 follows the current best Stage 3 result:
`recon=1.0, anchor=0.05 v3, bdsp=0.1, smooth=0.1 v4, r_tv=0.02`.

The latest results show that `r_tv=0.02` is the best balanced point, while
larger `r_tv` improves high-reference metrics but hurts self reconstruction.
These configs therefore do a narrow local search around `0.02` and isolate two
small regularizers that target the remaining R drift and overbright residuals.

LOLv2 high images are used only for validation checkpoint selection, not for
the training loss. All configs select checkpoints by `r_low_highref_psnr=max`.

## Experiment Matrix

| Exp | Purpose | Main loss change |
|---|---|---|
| A | Fine lower R_TV probe | `r_tv_weight=0.0175` |
| B | Fine upper R_TV probe | `r_tv_weight=0.0225` |
| C | Light R consistency isolation | `r_tv_weight=0.02`, `r_consistency_weight=0.01` |
| D | Light R saturation isolation | `r_tv_weight=0.02`, `r_sat_weight=0.0025` |

Expected readout:

- A/B decide whether the Stage 3 optimum is exactly at `0.02` or slightly
  shifted.
- C checks whether a much smaller consistency weight than the old `0.05` can
  reduce R drift without the previous quality drop.
- D checks whether saturation control can reduce the remaining overbright ratio
  without sacrificing the high-reference PSNR gains.
