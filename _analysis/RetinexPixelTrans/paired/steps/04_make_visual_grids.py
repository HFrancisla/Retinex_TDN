#!/usr/bin/env python3
"""Step 04: create selected visual grids from corrected ranking and details."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from paired_steps_common import (
    EXP_ROOT,
    FIG_ROOT,
    RESULT_ROOT,
    as_float,
    details_path,
    discover_runs,
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


def read_bgr(path: Path, gray: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    if gray:
        image = np.repeat(image[..., None], 3, axis=2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def heatmap(error: np.ndarray, scale: float = 0.25) -> np.ndarray:
    if error.ndim == 3:
        error = np.abs(error).mean(axis=2)
    scaled = np.clip(error / scale, 0, 1)
    colored = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_pil(image: np.ndarray, size: tuple[int, int]) -> Image.Image:
    array = np.clip(image * 255, 0, 255).astype(np.uint8)
    pil = Image.fromarray(array, mode="RGB")
    pil.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(pil, ((size[0] - pil.width) // 2, (size[1] - pil.height) // 2))
    return canvas


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


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
        draw.text((8, y + 45), label, fill="black", font=font(14))
        for col, image in enumerate(images[row]):
            canvas.paste(to_pil(image, (cell_w, cell_h)), (left + col * cell_w, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def dataset_split(config: dict) -> Path:
    data_root = Path(str(config.get("data", {}).get("path", "")))
    if not data_root.is_absolute():
        data_root = EXP_ROOT.parents[2] / data_root
    split = data_root / "val"
    if not split.is_dir():
        split = data_root / "test"
    return split


def load_run_rows(run_name: str, iteration: int) -> list[dict[str, str]]:
    rows = read_csv(details_path(EXP_ROOT / run_name))
    return [row for row in rows if int(float(row.get("iteration", -1))) == iteration]


def components(run_name: str, iteration: int, index: int, low_path: Path, high_path: Path) -> list[np.ndarray]:
    image_dir = EXP_ROOT / run_name / "img" / str(iteration)
    r_low = read_bgr(image_dir / f"{index}_R_low.png")
    r_high = read_bgr(image_dir / f"{index}_R_high.png")
    l_low = read_bgr(image_dir / f"{index}_L_low.png", gray=True)
    l_high = read_bgr(image_dir / f"{index}_L_high.png", gray=True)
    i_low = read_bgr(low_path)
    i_high = read_bgr(high_path)
    return [
        i_low,
        i_high,
        r_low,
        r_high,
        l_low,
        l_high,
        heatmap(r_low - r_high),
        heatmap(r_low - i_high),
    ]


def choose_cases(best_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    psnrs = sorted(as_float(row, "r_low_highref_psnr") for row in best_rows)
    median = psnrs[len(psnrs) // 2]
    typical = min(best_rows, key=lambda row: abs(as_float(row, "r_low_highref_psnr") - median))
    worst_abs = min(best_rows, key=lambda row: as_float(row, "r_low_highref_psnr"))
    worst_consistency = min(best_rows, key=lambda row: as_float(row, "r_consistency_psnr"))
    overbright = max(best_rows, key=lambda row: as_float(row, "r_low_highref_overbright_010", -math.inf))
    return [
        {"case": "typical_highref", "row": typical},
        {"case": "worst_highref", "row": worst_abs},
        {"case": "worst_consistency", "row": worst_consistency},
        {"case": "worst_overbright", "row": overbright},
    ]


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    ranking_path = RESULT_ROOT / "corrected_ranking.csv"
    if not ranking_path.is_file():
        raise SystemExit("Missing corrected_ranking.csv. Run 02_summarize_rank.py first.")
    ranking = read_csv(ranking_path)
    top_run_names = [row["run"] for row in ranking[: args.top_runs]]
    best_run = top_run_names[0]
    best_rows = load_run_rows(best_run, args.iteration)
    if not best_rows:
        raise SystemExit(f"No details rows for best run {best_run} at iteration {args.iteration}")
    cases = choose_cases(best_rows)

    first_config = run_config(EXP_ROOT / best_run)
    split = dataset_split(first_config)
    selected_rows = []
    for case in cases:
        row = case["row"]
        index = int(float(row["image_index"]))
        low_name = str(row.get("input_low_file", ""))
        high_name = str(row.get("input_high_file", ""))
        low_path = split / "low" / low_name
        high_path = split / "high" / high_name
        images = []
        row_labels = []
        for run_name in top_run_names:
            run_rows = {int(float(item["image_index"])): item for item in load_run_rows(run_name, args.iteration)}
            metrics = run_rows.get(index, {})
            images.append(components(run_name, args.iteration, index, low_path, high_path))
            row_labels.append(
                f"{run_name[:22]}\n"
                f"R→H {as_float(metrics, 'r_low_highref_psnr'):.1f}dB; "
                f"R/R {as_float(metrics, 'r_consistency_psnr'):.1f}dB"
            )
        output = FIG_ROOT / f"{case['case']}_index_{index}.png"
        make_grid(
            f"{case['case']} index={index}; heatmaps clipped at 0.25",
            row_labels,
            ["I low", "I high", "R low", "R high", "L low", "L high", "|Rlow-Rhigh|", "|Rlow-Ihigh|"],
            images,
            output,
        )
        selected_rows.append(
            {
                "case": case["case"],
                "best_run": best_run,
                "image_index": index,
                "input_low_file": low_name,
                "input_high_file": high_name,
                "r_low_highref_psnr": as_float(row, "r_low_highref_psnr"),
                "r_consistency_psnr": as_float(row, "r_consistency_psnr"),
                "overbright": as_float(row, "r_low_highref_overbright_010"),
                "figure": str(output),
            }
        )

    output_csv = RESULT_ROOT / "selected_visual_cases.csv"
    write_csv(output_csv, selected_rows)
    print(f"saved: {output_csv}")
    for row in selected_rows:
        print(f"saved: {row['figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
