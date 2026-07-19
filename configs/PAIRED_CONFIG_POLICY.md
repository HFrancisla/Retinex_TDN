# Paired Config Policy

For paired datasets with aligned high-reference images, such as LOLv2, all
`configs/*/paired/*.yaml` should keep the same evaluation and checkpoint
selection contract. Loss weights and model parameters may vary between
experiments; the fields below should not vary unless the validation protocol
changes intentionally.

```yaml
training:
  save_ckpt_interval: 500
  eval:
    eval_interval: 500
    save_img_interval: 0
    save_best_ckpt: true
    save_best_images: true
    selection_metric: "r_low_highref_psnr"
    selection_mode: "max"
    keep_top_ckpt: 3
    max_save_images: 100
    final_full_validation: false
```

Rationale:

- `r_low_highref_psnr` compares `R_low` against the paired high-reference image.
  This matches the goal of making the low-light reflectance look like the
  normal-light reference.
- `psnr` is only the old proxy between decomposed reflectance outputs, so it can
  prefer internally consistent but visually wrong decompositions.
- `total_loss` is a training objective mix and can favor reconstruction terms
  over the reflectance quality used for analysis.
- `save_img_interval: 0` avoids stale numeric folders such as `img/2000`.
  Training publishes the selected checkpoint once as `img/best`.
- `save_ckpt_interval: 500` plus `keep_top_ckpt: 3` keeps a small candidate pool
  for later full validation without keeping every checkpoint.
- Keep `final_full_validation: false` when only one published image set is
  desired. Enable it only for quick-validation experiments where an additional
  full-resolution pass and `img/final_best` output are intentional.
