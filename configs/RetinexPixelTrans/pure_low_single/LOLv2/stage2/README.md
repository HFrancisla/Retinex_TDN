# Stage 2 LOLv2 Pure-Low-Single Loss Matrix

This stage keeps the current strongest Stage 1 basin as the control:
`recon=1.0, anchor=0.05 v2, bdsp=0.1, smooth=0.1 v2`.

LOLv2 high images are used only for validation checkpoint selection, not for the
training loss. All configs therefore select checkpoints by
`r_low_highref_psnr=max` instead of `total_loss=min`. The configs use
`auto_name: true`; zero-valued optional new losses are explicitly declared and
kept in generated names for easier side-by-side comparison.

| Exp | Purpose | Main loss change |
|---|---|---|
| A | Current best control | baseline v2 anchor/smooth |
| B | Anchor/Smooth upgrade | `anchor_version=v3`, `smooth_version=v4` |
| C | Conservative R denoise | B + `r_tv_weight=0.01` |
| D | Stronger R denoise | B + `r_tv_weight=0.02` |
| E | R scale constraint | C + `r_consistency_weight=0.05` |
| F | Full soft constraint | E + `r_sat_weight=0.005` |

Compare C vs D to decide the TV strength. Compare E vs F to check whether the
saturation penalty helps over-bright R without hurting reconstruction.
