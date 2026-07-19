# Step 04 training dynamics

Eval rows parsed: `280`
Full-validation summaries: `5`

Use unweighted `full_recon_loss` when comparing runs with different `recon_weight`; weighted total loss is not directly comparable across recon weights.

## Recon-weight comparison warning

- `BDDnight_0.3r_0.05anchorv2_0.05bdsp_0.1smv1_20260716-102043`: full recon=0.00624, weighted recon=0.00187
- `BDDnight_0.3r_0.05anchorv2_0.05bdsp_0.5smv1_20260716-123043`: full recon=0.00900, weighted recon=0.00270

Figure: `/home/ipr4090/2024_hzf/Retinex_TDN/_analysis/RetinexPixelTrans/pure_low_single/steps/results/figures/training_loss_curves.png`
