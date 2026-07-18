#!/usr/bin/env python3
"""Step 05: build visual grids for pure-low-single diagnostic cases."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pure_single_steps_common import (
    EXP_ROOT,
    FIG_ROOT,
    RESULT_ROOT,
    as_float,
    dataset_input_roots,
    details_path,
    ensure_output_dirs,
    read_csv,
    run_config,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--top-runs", type=int, default=5)
    return parser.parse_args()


def read_rgb(path: Path, gray: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    if gray:
        image = np.repeat(image[..., None], 3, axis=2)
        return image.astype(np.float32) / 255.0
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def heatmap(error: np.ndarray, scale: float = 0.35) -> np.ndarray:
    if error.ndim == 3:
        error = np.abs(error).mean(axis=2)
    scaled = np.clip(error / scale, 0, 1)
    colored = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def to_pil(image: np.ndarray, size: tuple[int, int]) -> Image.Image:
    array = np.clip(image * 255, 0, 255).astype(np.uint8)
    pil = Image.fromarray(array, mode="RGB")
    pil.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(pil, ((size[0] - pil.width) // 2, (size[1] - pil.height) // 2))
    return canvas


def make_grid(title: str, row_labels: list[str], col_labels: list[str], images: list[list[np.ndarray]], output: Path) -> None:
    cell_w, cell_h = 210, 140
    left, top = 190, 78
    canvas = Image.new("RGB", (left + cell_w * len(col_labels), top + cell_h * len(row_labels)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), title, fill="black", font=font(22))
    for col, label in enumerate(col_labels):
        draw.text((left + col * cell_w + 6, 50), label, fill="black", font=font(15))
    for row, label in enumerate(row_labels):
        y = top + row * cell_h
        draw.text((8, y + 40), label, fill="black", font=font(14))
        for col, image in enumerate(images[row]):
            canvas.paste(to_pil(image, (cell_w, cell_h)), (left + col * cell_w, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def run_rows(run_name: str, iteration: int) -> list[dict[str, str]]:
    rows = read_csv(details_path(EXP_ROOT / run_name))
    return [row for row in rows if int(float(row.get("iteration", -1))) == iteration]


def choose_cases(rows: list[dict[str, str]], has_high: bool) -> list[tuple[str, dict[str, str]]]:
    psnrs = sorted(as_float(row, "self_low_psnr") for row in rows)
    median = psnrs[len(psnrs) // 2]
    cases = [
        ("typical", min(rows, key=lambda row: abs(as_float(row, "self_low_psnr") - median))),
        ("worst_recon", min(rows, key=lambda row: as_float(row, "self_low_psnr"))),
        ("worst_noise", max(rows, key=lambda row: as_float(row, "r_low_tv_to_input", -math.inf))),
        ("worst_l_leakage", max(rows, key=lambda row: as_float(row, "l_low_input_gray_corr", -math.inf))),
    ]
    if has_high and any("r_low_highref_psnr" in row for row in rows):
        cases.append(("worst_highref", min(rows, key=lambda row: as_float(row, "r_low_highref_psnr"))))
    return cases


def components(run_name: str, iteration: int, index: int, low_path: Path, high_path: Path | None) -> list[np.ndarray]:
    image_dir = EXP_ROOT / run_name / "img" / str(iteration)
    i_low = read_rgb(low_path)
    r = read_rgb(image_dir / f"{index}_R_low.png")
    l = read_rgb(image_dir / f"{index}_L_low.png", gray=True)
    synthesis = np.clip(r * l[..., :1], 0, 1)
    recon_err = heatmap(synthesis - i_low)
    if high_path is not None and high_path.is_file():
        i_high = read_rgb(high_path)
        return [i_low, i_high, r, l, synthesis, recon_err, heatmap(r - i_high)]
    return [i_low, r, l, synthesis, recon_err]


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    ranking_path = RESULT_ROOT / "pure_single_ranking.csv"
    if not ranking_path.is_file():
        raise SystemExit("Missing pure_single_ranking.csv. Run 02_summarize_rank.py first.")
    ranking = read_csv(ranking_path)
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ranking:
        by_dataset[row["dataset"]].append(row)

    selected = []
    for dataset, ranked in sorted(by_dataset.items()):
        top = ranked[: args.top_runs]
        best = top[0]
        best_rows = run_rows(best["run"], args.iteration)
        if not best_rows:
            continue
        config = run_config(EXP_ROOT / best["run"])
        low_root, high_root = dataset_input_roots(config, best["run"])
        has_high = high_root is not None and high_root.is_dir()
        low_files = sorted(low_root.glob("*.*"))
        high_files = sorted(high_root.glob("*.*")) if has_high else []
        for case_name, case_row in choose_cases(best_rows, has_high):
            index = int(float(case_row["image_index"]))
            low_name = str(case_row.get("input_low_file", "")) or low_files[index].name
            high_name = str(case_row.get("input_high_file", ""))
            low_path = low_root / low_name
            high_path = (high_root / high_name) if high_root is not None and high_name else (high_files[index] if high_files else None)
            images = []
            row_labels = []
            for run in top:
                metrics = {int(float(row["image_index"])): row for row in run_rows(run["run"], args.iteration)}.get(index, {})
                images.append(components(run["run"], args.iteration, index, low_path, high_path))
                row_labels.append(
                    f"{run['label']}\n"
                    f"self {as_float(metrics, 'self_low_psnr'):.1f}dB; "
                    f"RTV {as_float(metrics, 'r_low_tv_to_input'):.1f}"
                )
            cols = ["I low", "I high", "R", "L", "R×L", "|S-I|", "|R-high|"] if has_high else ["I low", "R", "L", "R×L", "|S-I|"]
            output = FIG_ROOT / f"{dataset.lower()}_{case_name}_index_{index}.png"
            make_grid(f"{dataset} {case_name} index={index}; heatmaps clipped", row_labels, cols, images, output)
            selected.append(
                {
                    "dataset": dataset,
                    "case": case_name,
                    "best_run": best["run"],
                    "image_index": index,
                    "input_low_file": low_name,
                    "input_high_file": high_name,
                    "self_low_psnr": as_float(case_row, "self_low_psnr"),
                    "r_low_tv_to_input": as_float(case_row, "r_low_tv_to_input"),
                    "l_low_input_gray_corr": as_float(case_row, "l_low_input_gray_corr"),
                    "r_low_highref_psnr": as_float(case_row, "r_low_highref_psnr"),
                    "figure": str(output),
                }
            )

    write_csv(RESULT_ROOT / "selected_visual_cases.csv", selected)
    print(f"saved: {RESULT_ROOT / 'selected_visual_cases.csv'}")
    for row in selected:
        print(f"saved: {row['figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
