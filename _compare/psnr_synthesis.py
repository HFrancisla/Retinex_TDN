#!/usr/bin/env python3
"""Measure saved ``S = R * L`` reconstruction against the actual inputs.

This is a reconstruction-integrity check only.  It must never be interpreted as
reflectance quality.  Paired runs report both low- and high-domain synthesis;
pure-low-single runs report low-domain synthesis using the exact validation
ordering/layout resolved from config.yaml.
"""

from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_decomposition import (
    align_input,
    indexed_paths,
    psnr,
    records_for_iteration,
    resolve_validation_records,
)


ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = Path(os.environ.get("RETINEX_SYNTH_EXP_DIR", ROOT / "experiments"))
FORCE = os.environ.get("RETINEX_SYNTH_FORCE", "0") == "1"


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read image: {path}")
    return image.astype(np.float32) / 255.0


def summarize(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def report_is_fresh(report: Path, sources: list[Path]) -> bool:
    if FORCE or not report.is_file():
        return False
    expected = f"# source_signature: {sources_signature(sources)}"
    return expected in report.read_text(encoding="utf-8").splitlines()[:5]


def sources_signature(sources: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in sources}):
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def process_run(run_dir: Path) -> None:
    config_path = run_dir / "config.yaml"
    synthesis_root = run_dir / "synthesis"
    if not config_path.is_file() or not synthesis_root.is_dir():
        return
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    mode = str(config.get("data", {}).get("mode", "unknown"))
    records = resolve_validation_records(run_dir, config)
    iteration_dirs = sorted(
        (path for path in synthesis_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    report = run_dir / "synthesis_compare.txt"
    sources = [Path(__file__), SCRIPT_DIR / "analyze_decomposition.py", config_path]
    sources.extend(
        path for path in (
            run_dir / "quick_val_manifest.txt",
            run_dir / "final_full_validation.yaml",
        ) if path.is_file()
    )
    for iteration_dir in iteration_dirs:
        sources.extend(iteration_dir.glob("*.png"))
        image_dir = run_dir / "img" / iteration_dir.name
        if image_dir.is_dir():
            sources.extend(image_dir.glob("*.png"))
    sources.extend(record.low for record in records)
    sources.extend(record.high for record in records if record.high is not None)
    if report_is_fresh(report, sources):
        print(f"  [SKIP] {report.name} 已是最新")
        return

    rows: list[dict[str, int | float]] = []
    for iteration_dir in iteration_dirs:
        image_dir = run_dir / "img" / iteration_dir.name
        if not image_dir.is_dir():
            raise FileNotFoundError(f"synthesis has no matching img directory: {image_dir}")
        iteration_records = records_for_iteration(run_dir, image_dir, records)
        r_low_indices = set(indexed_paths(image_dir, "R_low"))
        l_low_indices = set(indexed_paths(image_dir, "L_low"))
        if r_low_indices != l_low_indices:
            raise ValueError(
                f"iteration {iteration_dir.name} R_low/L_low indices differ: "
                f"R-only={sorted(r_low_indices-l_low_indices)[:10]}, "
                f"L-only={sorted(l_low_indices-r_low_indices)[:10]}"
            )
        synthesis_low_paths = indexed_paths(iteration_dir, "S_low")
        if set(synthesis_low_paths) != r_low_indices:
            raise ValueError(
                f"iteration {iteration_dir.name} S_low is incomplete/stale: "
                f"missing={sorted(r_low_indices-set(synthesis_low_paths))[:10]}, "
                f"extra={sorted(set(synthesis_low_paths)-r_low_indices)[:10]}"
            )

        r_high_indices = set(indexed_paths(image_dir, "R_high"))
        l_high_indices = set(indexed_paths(image_dir, "L_high"))
        if r_high_indices != l_high_indices:
            raise ValueError(
                f"iteration {iteration_dir.name} R_high/L_high indices differ"
            )
        synthesis_high_paths = indexed_paths(iteration_dir, "S_high")
        if set(synthesis_high_paths) != r_high_indices:
            raise ValueError(
                f"iteration {iteration_dir.name} S_high is incomplete/stale: "
                f"missing={sorted(r_high_indices-set(synthesis_high_paths))[:10]}, "
                f"extra={sorted(set(synthesis_high_paths)-r_high_indices)[:10]}"
            )
        low_psnr: list[float] = []
        low_l1: list[float] = []
        high_psnr: list[float] = []
        high_l1: list[float] = []
        for index, synthesis_path in sorted(synthesis_low_paths.items()):
            if index >= len(iteration_records):
                raise ValueError(f"synthesis index {index} exceeds validation records")
            synthesis = read_color(synthesis_path)
            record = iteration_records[index]
            target = align_input(read_color(record.low), synthesis.shape[:2], record.center_crop)
            low_psnr.append(psnr(synthesis, target))
            low_l1.append(float(np.abs(synthesis - target).mean()))
        for index, synthesis_path in sorted(synthesis_high_paths.items()):
            if index >= len(iteration_records) or iteration_records[index].high is None:
                raise ValueError(f"S_high index {index} has no matched high validation image")
            synthesis = read_color(synthesis_path)
            record = iteration_records[index]
            target = align_input(
                read_color(record.high), synthesis.shape[:2], record.center_crop
            )
            high_psnr.append(psnr(synthesis, target))
            high_l1.append(float(np.abs(synthesis - target).mean()))
        if not low_psnr:
            continue
        low_mean, low_std = summarize(low_psnr)
        row: dict[str, int | float] = {
            "iteration": int(iteration_dir.name), "n_low": len(low_psnr),
            "low_psnr_mean": low_mean, "low_psnr_std": low_std,
            "low_l1_mean": float(np.mean(low_l1)),
            "n_high": len(high_psnr),
        }
        if high_psnr:
            high_mean, high_std = summarize(high_psnr)
            row.update({
                "high_psnr_mean": high_mean, "high_psnr_std": high_std,
                "high_l1_mean": float(np.mean(high_l1)),
            })
        rows.append(row)

    lines = [
        f"# {run_dir.name}", f"# mode: {mode}",
        f"# source_signature: {sources_signature(sources)}",
        "# Reconstruction integrity only: high PSNR does not imply correct R/L semantics.", "",
        f"{'iter':>10}  {'n_low':>6}  {'Slo_PSNR':>10}  {'std':>8}  {'Slo_L1':>9}  "
        f"{'n_high':>7}  {'Shi_PSNR':>10}  {'std':>8}  {'Shi_L1':>9}",
        f"{'-'*10}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*9}  "
        f"{'-'*7}  {'-'*10}  {'-'*8}  {'-'*9}",
    ]
    for row in rows:
        def value(key: str, spec: str, width: int) -> str:
            item = row.get(key)
            return f"{item:{width}{spec}}" if item is not None else f"{'-':>{width}}"
        lines.append(
            f"{row['iteration']:10d}  {row['n_low']:6d}  "
            f"{row['low_psnr_mean']:10.2f}  {row['low_psnr_std']:8.2f}  "
            f"{row['low_l1_mean']:9.5f}  {row['n_high']:7d}  "
            f"{value('high_psnr_mean', '.2f', 10)}  {value('high_psnr_std', '.2f', 8)}  "
            f"{value('high_l1_mean', '.5f', 9)}"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if rows:
        print(f"  -> {report.name} S_low PSNR={rows[-1]['low_psnr_mean']:.2f} dB")


def main() -> None:
    runs = []
    for model_dir in sorted(EXP_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for mode_dir in sorted(model_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            for run_dir in sorted(mode_dir.iterdir()):
                if (run_dir / "synthesis").exists() and (run_dir / "config.yaml").exists():
                    runs.append(run_dir)
    print(f"{len(runs)} runs")
    for index, run in enumerate(runs, 1):
        print(f"[{index}/{len(runs)}] {run.relative_to(EXP_DIR)}")
        process_run(run)
    print("done")


if __name__ == "__main__":
    main()
