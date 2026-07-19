#!/usr/bin/env python3
"""Step 03: compare each run against the per-dataset conservative baseline."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import cv2
import numpy as np

from pure_single_steps_common import (
    EXP_ROOT,
    RESULT_ROOT,
    add_image_set_args,
    as_float,
    detail_rows_for_image_set,
    details_path,
    discover_runs,
    ensure_output_dirs,
    markdown_table,
    read_csv,
    resolve_image_set,
    run_config,
    run_label,
    selected_image_set,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_image_set_args(parser)
    return parser.parse_args()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def read_image(path, gray: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32) / 255.0


def baseline_for(dataset: str, runs: list[dict]) -> dict | None:
    candidates = []
    for row in runs:
        config = run_config(EXP_ROOT / row["run"])
        loss = config.get("loss", {}) or {}
        if (
            float(loss.get("recon_weight", -1)) == 1.0
            and str(loss.get("anchor_version", "")) == "v2"
            and float(loss.get("smooth_weight", -1)) == 0.0
        ):
            candidates.append(row)
    return candidates[0] if candidates else None


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    image_set = selected_image_set(args)
    run_rows = []
    for run_dir in discover_runs():
        details = details_path(run_dir)
        if not details.is_file():
            continue
        config = run_config(run_dir)
        resolved_image_set = resolve_image_set(run_dir, image_set)
        rows = detail_rows_for_image_set(read_csv(details), args, resolved_image_set)
        if not rows:
            continue
        run_rows.append(
            {
                "run": run_dir.name,
                "label": run_label(run_dir, config),
                "dataset": str((config.get("data", {}) or {}).get("path", "")),
                "image_set": resolved_image_set,
                "details_rows": rows,
            }
        )

    # Normalize dataset name from summary/ranking if available.
    ranking_path = RESULT_ROOT / "pure_single_ranking.csv"
    if ranking_path.is_file():
        by_run = {row["run"]: row for row in read_csv(ranking_path)}
        for row in run_rows:
            if row["run"] in by_run:
                row["dataset"] = by_run[row["run"]]["dataset"]

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in run_rows:
        by_dataset[str(row["dataset"])].append(row)

    output_rows = []
    for dataset, rows in sorted(by_dataset.items()):
        baseline = baseline_for(dataset, rows)
        if baseline is None:
            continue
        baseline_dir = EXP_ROOT / baseline["run"] / "img" / baseline["image_set"]
        baseline_details = {int(float(row["image_index"])): row for row in baseline["details_rows"]}
        for row in rows:
            run_dir = EXP_ROOT / row["run"] / "img" / row["image_set"]
            values: dict[str, list[float]] = defaultdict(list)
            for detail in row["details_rows"]:
                index = int(float(detail["image_index"]))
                if index not in baseline_details:
                    continue
                r_base = read_image(baseline_dir / f"{index}_R_low.png")
                l_base = read_image(baseline_dir / f"{index}_L_low.png", gray=True)
                r = read_image(run_dir / f"{index}_R_low.png")
                l = read_image(run_dir / f"{index}_L_low.png", gray=True)
                values["r_l1"].append(float(np.abs(r - r_base).mean()))
                values["r_psnr"].append(psnr(r, r_base))
                values["l_l1"].append(float(np.abs(l - l_base).mean()))
                values["r_mean_delta"].append(as_float(detail, "r_low_mean") - as_float(baseline_details[index], "r_low_mean"))
                values["r_tv_to_input_delta"].append(as_float(detail, "r_low_tv_to_input") - as_float(baseline_details[index], "r_low_tv_to_input"))
                values["l_corr_delta"].append(as_float(detail, "l_low_input_gray_corr") - as_float(baseline_details[index], "l_low_input_gray_corr"))
            out = {
                "dataset": dataset,
                "baseline_run": baseline["run"],
                "baseline_label": baseline["label"],
                "run": row["run"],
                "label": row["label"],
                "n": len(values["r_l1"]),
            }
            for key, vals in values.items():
                finite = [value for value in vals if math.isfinite(value)]
                out[f"{key}_mean"] = sum(finite) / len(finite) if finite else math.nan
                out[f"{key}_p95"] = float(np.quantile(finite, 0.95)) if finite else math.nan
            output_rows.append(out)

    write_csv(RESULT_ROOT / "cross_run_stability.csv", output_rows)
    md_lines = [
        "# Step 03 cross-run stability",
        "",
        "Baseline per dataset: `recon=1.0, anchor=v2, smooth=0`.",
        "",
    ]
    for dataset in sorted({row["dataset"] for row in output_rows}):
        rows = [row for row in output_rows if row["dataset"] == dataset]
        md_lines.extend([
            f"## {dataset}",
            "",
            markdown_table(
                rows,
                [
                    ("run", "run", ""),
                    ("label", "label", ""),
                    ("R L1", "r_l1_mean", ".4f"),
                    ("R PSNR", "r_psnr_mean", ".2f"),
                    ("L L1", "l_l1_mean", ".4f"),
                    ("ΔR mean", "r_mean_delta_mean", ".4f"),
                    ("ΔR TV/input", "r_tv_to_input_delta_mean", ".2f"),
                    ("Δcorr(L,I)", "l_corr_delta_mean", ".3f"),
                ],
            ),
            "",
        ])
    (RESULT_ROOT / "cross_run_stability.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"saved: {RESULT_ROOT / 'cross_run_stability.csv'}")
    print(f"saved: {RESULT_ROOT / 'cross_run_stability.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
