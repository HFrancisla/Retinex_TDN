#!/usr/bin/env python3
"""Publish ``weights/best_model.pth`` outputs to ``img/best``.

This is intended for historic runs that only kept numeric visualization
directories.  It renders the best checkpoint again, writes image-set metadata,
generates ``synthesis/best``, and optionally moves old ``img/<number>`` dirs
out of ``img/`` so the run has a single canonical image set.
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import shutil
import sys
from pathlib import Path

import torch
from torchvision import transforms as torchvision_T
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import MyDataSet, PureLowSingleDataSet, transforms as T
from models import (
    RetinexPixelClassic,
    RetinexPixelTrans,
    RetinexPixelTransMinus,
    RetinexPointRaw,
)
from utils import _build_loss_function, evaluate, load_config, read_data, read_pure_low_data
from _compare.synthesize_retinex import process_iteration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help="run directory, or path relative to experiments/; repeatable",
    )
    parser.add_argument(
        "--stage4-paired",
        action="store_true",
        help="publish all RetinexPixelTrans paired runs created in the stage4 batch",
    )
    parser.add_argument("--image-set", default="best")
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--max-save-images", type=int, default=None)
    parser.add_argument(
        "--keep-old-img",
        action="store_true",
        help="do not move old numeric img directories to img_archived/",
    )
    return parser.parse_args()


def build_model(model_cfg: dict, device: torch.device):
    name = model_cfg.get("name", "RetinexPointRaw")
    if name == "RetinexPixelClassic":
        model = RetinexPixelClassic(
            dim=model_cfg.get("dim", 24),
            l_channel=model_cfg.get("l_channel", 32),
        )
    elif name == "RetinexPixelTrans":
        model = RetinexPixelTrans(
            dim=model_cfg.get("dim", 24),
            l_heads=model_cfg.get("l_heads", 1),
            l_ffn_expansion=model_cfg.get("l_ffn_expansion", 2.66),
        )
    elif name == "RetinexPixelTransMinus":
        model = RetinexPixelTransMinus(
            dim=model_cfg.get("dim", 24),
            l_heads=model_cfg.get("l_heads", 1),
            l_ffn_expansion=model_cfg.get("l_ffn_expansion", 2.66),
        )
    elif name == "RetinexPointRaw":
        model = RetinexPointRaw(dim=model_cfg.get("dim", 24))
    else:
        raise ValueError(f"unknown model.name={name!r}")
    return model.to(device)


def fixed_validation_subset(paths, size: int, seed: int):
    if size <= 0 or size >= len(paths):
        indices = list(range(len(paths)))
    else:
        indices = sorted(random.Random(seed).sample(range(len(paths)), size))
    return [paths[index] for index in indices], indices


def validation_loader(data_cfg: dict, eval_cfg: dict, training_cfg: dict, device: torch.device):
    data_root = Path(str(data_cfg.get("path", ""))).expanduser()
    if not data_root.is_absolute():
        data_root = (ROOT / data_root).resolve()
    data_mode = data_cfg["mode"]
    if data_mode == "pure_low_single":
        _, val_low_path, val_high_path = read_pure_low_data(
            str(data_root), include_val_high_ref=True
        )
    else:
        _, _, val_low_path, val_high_path = read_data(str(data_root), mode=data_mode)

    quick_val_size = int(eval_cfg.get("quick_val_size", 0) or 0)
    quick_val_crop_size = int(eval_cfg.get("quick_val_crop_size", 0) or 0)
    seed = int(training_cfg.get("seed", 0) or 0)
    quick_val_enabled = quick_val_size > 0
    transform_label = "quick_validation" if quick_val_enabled else "validation"

    if quick_val_enabled:
        val_low_path, quick_indices = fixed_validation_subset(
            list(val_low_path), quick_val_size, seed
        )
        if data_mode in ("paired", "pure_low_single") and val_high_path:
            val_high_path = [list(val_high_path)[index] for index in quick_indices]
        elif val_high_path:
            val_high_path, _ = fixed_validation_subset(
                list(val_high_path), quick_val_size, seed + 1
            )

    if data_mode == "pure_low_single":
        if val_high_path:
            transforms = []
            if quick_val_enabled and quick_val_crop_size > 0:
                transforms.append(T.CenterCrop(quick_val_crop_size))
            transforms.append(T.ToTensor())
            dataset = MyDataSet(val_low_path, val_high_path, transform=T.Compose(transforms))
        else:
            transforms = []
            if quick_val_enabled and quick_val_crop_size > 0:
                transforms.append(torchvision_T.CenterCrop(quick_val_crop_size))
            transforms.append(torchvision_T.ToTensor())
            dataset = PureLowSingleDataSet(
                val_low_path,
                transform=torchvision_T.Compose(transforms),
            )
    else:
        transforms = []
        if quick_val_enabled and quick_val_crop_size > 0:
            transforms.append(T.CenterCrop(quick_val_crop_size))
        transforms.append(T.ToTensor())
        dataset = MyDataSet(val_low_path, val_high_path, transform=T.Compose(transforms))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(data_cfg.get("val_batch_size", 1)),
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    return loader, transform_label


def resolve_run(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path, ROOT / "experiments" / value]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"experiment not found: {value}")


def default_stage4_runs() -> list[Path]:
    root = ROOT / "experiments" / "RetinexPixelTrans" / "paired"
    runs = []
    for run_dir in sorted(root.glob("*20260719-*")):
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.is_file():
            continue
        cfg = load_config(str(cfg_path))
        if cfg.get("data", {}).get("mode") == "paired":
            runs.append(run_dir.resolve())
    return runs


def archive_old_image_sets(run_dir: Path, image_set: str) -> list[str]:
    img_root = run_dir / "img"
    archived: list[str] = []
    if not img_root.is_dir():
        return archived
    old_dirs = [
        path for path in img_root.iterdir()
        if path.is_dir() and path.name != image_set and path.name.isdigit()
    ]
    if not old_dirs:
        return archived
    archive_root = run_dir / "img_archived" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_root.mkdir(parents=True, exist_ok=True)
    for old_dir in sorted(old_dirs, key=lambda path: int(path.name)):
        target = archive_root / old_dir.name
        shutil.move(str(old_dir), str(target))
        archived.append(str(target.relative_to(run_dir)))
    return archived


def publish_one(run_dir: Path, args: argparse.Namespace, device: torch.device) -> dict:
    cfg = load_config(str(run_dir / "config.yaml"))
    checkpoint_path = run_dir / "weights" / args.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(cfg.get("model", {}), device)
    model.load_state_dict(checkpoint["model"], strict=True)

    loss_function = _build_loss_function(cfg.get("loss", {})).to(device)
    training_cfg = cfg.get("training", {})
    eval_cfg = training_cfg.get("eval", {})
    loader, transform_label = validation_loader(
        cfg.get("data", {}), eval_cfg, training_cfg, device
    )
    max_save_images = args.max_save_images
    if max_save_images is None:
        max_save_images = int(
            cfg.get("training", {}).get("eval", {}).get("max_save_images", len(loader.dataset))
        )

    temp_root = run_dir / "img" / ".publish_best_tmp"
    shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    metrics, psnr = evaluate(
        model=model,
        data_loader=loader,
        device=device,
        lr=0.0,
        filefold_path=str(temp_root),
        loss_function=loss_function,
        save_images=True,
        global_iter=args.image_set,
        max_save_images=max_save_images,
    )

    rendered_dir = temp_root / args.image_set
    target_dir = run_dir / "img" / args.image_set
    if target_dir.is_dir():
        shutil.rmtree(target_dir)
    os.replace(rendered_dir, target_dir)
    shutil.rmtree(temp_root, ignore_errors=True)

    synth_dir = run_dir / "synthesis" / args.image_set
    if synth_dir.is_dir():
        shutil.rmtree(synth_dir)
    synth_stats = process_iteration(target_dir, synth_dir)

    archived = [] if args.keep_old_img else archive_old_image_sets(run_dir, args.image_set)
    selection_metric = checkpoint.get(
        "selection_metric",
        cfg.get("training", {}).get("eval", {}).get("selection_metric"),
    )
    selection_mode = checkpoint.get(
        "selection_mode",
        cfg.get("training", {}).get("eval", {}).get("selection_mode"),
    )
    selection_value = psnr if selection_metric == "psnr" else metrics.get(selection_metric)
    metadata = {
        "image_set": args.image_set,
        "checkpoint": str(checkpoint_path.relative_to(run_dir)),
        "checkpoint_step": int(checkpoint.get("global_iter", -1)),
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "selection_value": selection_value,
        "metrics": metrics,
        "psnr": psnr,
        "sample_count": len(loader.dataset),
        "max_save_images": max_save_images,
        "transform": transform_label,
        "published_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "synthesis_dir": str(synth_dir.relative_to(run_dir)),
        "synthesis_stats": synth_stats,
        "archived_old_img_dirs": archived,
    }
    (target_dir / "_image_set.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    args = parse_args()
    if args.stage4_paired:
        runs = default_stage4_runs()
    else:
        runs = [resolve_run(value) for value in args.experiments or []]
    if not runs:
        raise SystemExit("No experiments selected. Use --stage4-paired or --experiment.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"runs: {len(runs)}")
    failures = 0
    for index, run_dir in enumerate(runs, start=1):
        rel = run_dir.relative_to(ROOT / "experiments")
        print(f"\n[{index}/{len(runs)}] {rel}")
        try:
            metadata = publish_one(run_dir, args, device)
            print(
                f"published img/{args.image_set} from {args.checkpoint} "
                f"step={metadata['checkpoint_step']} "
                f"archived={len(metadata['archived_old_img_dirs'])}"
            )
        except Exception as error:
            failures += 1
            print(f"[ERROR] {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
