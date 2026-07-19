# Pure-Low-Single Config Policy

`pure_low_single` training always uses only low-light images for the loss. When
the validation dataset also has aligned high-reference images, the high images
may be used for checkpoint selection only; they are not used by the training
loss.

## LOLv2

LOLv2 has aligned `test/low` and `test/high` images, so use the reference-based
metric for checkpoint selection:

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

`r_low_highref_psnr` checks whether `R_low` is close to the paired high-light
reference image. This is a validation-only metric and better matches the desired
reflectance quality than `recon_loss`.

## BDD

BDDnight does not have aligned high-reference images, so do not use
`r_low_highref_psnr`. Use the configured unsupervised loss for per-run checkpoint
selection, and use final full validation because the quick validation set is a
fixed subset/crop of a much larger validation split:

```yaml
training:
  save_ckpt_interval: 500
  eval:
    eval_interval: 500
    save_img_interval: 0
    save_best_ckpt: true
    save_best_images: true
    selection_metric: "total_loss"
    selection_mode: "min"
    keep_top_ckpt: 3
    max_save_images: 100
    quick_val_size: 1000
    quick_val_crop_size: 512
    final_full_validation: true
    final_val_batch_size: 2
```

For BDD, `total_loss` is suitable for selecting checkpoints within one run
because it includes reconstruction, anchor, BDSP, and smooth terms according to
that run's fixed weights. It should not be used as the only cross-run ranking
metric when loss weights differ; cross-run analysis should compare component
metrics, visual outputs, and downstream behavior.

`final_full_validation: true` publishes `img/final_best` in addition to
`img/best`. This is intentional for BDD: `img/best` comes from quick validation,
while `img/final_best` is selected from top checkpoints on the full validation
set at full resolution.
