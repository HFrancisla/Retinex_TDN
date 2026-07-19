#!/usr/bin/env python3
"""Analyze saved Retinex decompositions with mode-appropriate metrics.

The report deliberately separates three questions:

* reconstruction: does ``R * L`` reproduce the input?
* paired consistency: are ``R_low`` and ``R_high`` alike?
* high-reference fidelity: is ``R_low`` close to the matched normal-light image?

The last metric is available for paired data and, as an explicitly diagnostic
reference, for pure-low-single data whose validation low/high filenames match
exactly.  A normal-light photograph is not claimed to be reflectance ground
truth.  Pure-low datasets without matched high images remain fully supported
through no-reference decomposition diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import yaml
from skimage.metrics import structural_similarity


DEFAULT_PREFIX = "decomposition_analysis"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANALYSIS_VERSION = 2
PREFERRED_IMAGE_SETS = ("final_best", "best")


def image_set_name_sort_key(name: str):
    if name in PREFERRED_IMAGE_SETS:
        return (0, PREFERRED_IMAGE_SETS.index(name), 0, "")
    if name.isdigit():
        return (1, 0, int(name), "")
    return (2, 0, 0, name)


def is_selectable_image_set(name: str) -> bool:
    return name in PREFERRED_IMAGE_SETS or name.isdigit()


class InputRecord(NamedTuple):
    low: Path
    high: Path | None = None
    center_crop: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="experiment run directory or relative run name")
    parser.add_argument(
        "--experiments-dir", type=Path, default=Path("experiments"),
        help="root used when experiment is a relative run name",
    )
    parser.add_argument(
        "--iteration", type=int, action="append",
        help="analyze only this numeric iteration; repeatable (default: final_best if present, otherwise best)",
    )
    parser.add_argument(
        "--image-set", action="append",
        help="analyze this img image-set directory by name, e.g. best, final_best, or 2000",
    )
    parser.add_argument(
        "--all-image-sets", action="store_true",
        help="analyze every available img image set, including named sets and numeric iterations",
    )
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--details", action="store_true", help="write per-image CSV")
    parser.add_argument("--force", action="store_true", help="ignore freshness cache")
    return parser.parse_args()


def resolve_experiment(value: str, experiments_dir: Path) -> Path:
    direct = Path(value).expanduser()
    for candidate in (direct, experiments_dir.expanduser() / value):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "experiment directory not found; tried: "
        + ", ".join(str(path) for path in (direct, experiments_dir / value))
    )


def load_config(experiment_dir: Path) -> dict:
    path = experiment_dir / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing config.yaml: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def find_iteration_dirs(
    experiment_dir: Path,
    selected: list[int] | None,
    selected_image_sets: list[str] | None = None,
    all_image_sets: bool = False,
) -> tuple[list[Path], list[str]]:
    root = experiment_dir / "img"
    warnings: list[str] = []
    if not root.is_dir():
        raise FileNotFoundError(f"missing img directory: {root}")
    directories = {
        int(path.name): path for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    }
    named_dirs = {
        path.name: path for path in root.iterdir()
        if path.is_dir() and not path.name.isdigit() and is_selectable_image_set(path.name)
    }
    if not directories and not named_dirs:
        raise FileNotFoundError(f"no image-set directory below: {root}")

    def preferred_named_dir() -> Path | None:
        if "final_best" in named_dirs:
            return named_dirs["final_best"]
        if "best" in named_dirs:
            return named_dirs["best"]
        final_report = experiment_dir / "final_full_validation.yaml"
        if final_report.is_file():
            report = yaml.safe_load(final_report.read_text(encoding="utf-8")) or {}
            published = report.get("published_image_dir")
            if published:
                candidate = experiment_dir / str(published)
                if candidate.is_dir():
                    return candidate
        return None

    if selected_image_sets:
        result_dirs: list[Path] = []
        for name in dict.fromkeys(selected_image_sets):
            if name.isdigit() and int(name) in directories:
                result_dirs.append(directories[int(name)])
            elif name in named_dirs:
                result_dirs.append(named_dirs[name])
            else:
                raise FileNotFoundError(f"requested image set {name!r} not found below: {root}")
        return result_dirs, warnings

    if selected:
        result_dirs = []
        available = sorted(directories.keys())
        for value in sorted(set(selected)):
            if value in directories:
                result_dirs.append(directories[value])
            elif preferred := preferred_named_dir():
                warn_msg = (
                    f"requested iteration {value} not found for {experiment_dir.name}, "
                    f"using published image set: {preferred.name}"
                )
                print(f"[WARN] {warn_msg}", file=sys.stderr)
                warnings.append(warn_msg)
                result_dirs.append(preferred)
            else:
                if not available:
                    raise FileNotFoundError(
                        f"requested iteration {value} not found and no numeric fallback exists: {root}"
                    )
                fallback_val = min(available, key=lambda x: abs(x - value))
                warn_msg = (
                    f"requested iteration {value} not found for {experiment_dir.name}, "
                    f"falling back to closest available: {fallback_val}"
                )
                print(f"[WARN] {warn_msg}", file=sys.stderr)
                warnings.append(warn_msg)
                result_dirs.append(directories[fallback_val])
        # Deduplicate while preserving order
        unique_dirs = []
        for d in result_dirs:
            if d not in unique_dirs:
                unique_dirs.append(d)
        return unique_dirs, warnings

    if all_image_sets:
        ordered = [named_dirs[name] for name in sorted(named_dirs, key=image_set_name_sort_key)]
        ordered.extend(directories[value] for value in sorted(directories))
        return ordered, warnings

    if preferred := preferred_named_dir():
        return [preferred], warnings
    return [directories[value] for value in sorted(directories)], warnings


def iteration_value(iteration_dir: Path) -> int:
    if iteration_dir.name.isdigit():
        return int(iteration_dir.name)
    metadata_path = iteration_dir / "_image_set.yaml"
    if metadata_path.is_file():
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        checkpoint_step = metadata.get("checkpoint_step")
        if checkpoint_step is not None:
            return int(checkpoint_step)
    raise ValueError(
        f"non-numeric image directory needs _image_set.yaml with checkpoint_step: "
        f"{iteration_dir}"
    )


def _image_files(directory: Path, required: bool = True) -> list[Path]:
    if not directory.is_dir():
        if required:
            raise FileNotFoundError(f"image directory does not exist: {directory}")
        return []
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if required and not files:
        raise ValueError(f"image directory is empty: {directory}")
    return files


def _pair_by_stem(low_files: list[Path], high_files: list[Path]) -> list[InputRecord]:
    def index(paths: list[Path], domain: str) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for path in paths:
            if path.stem in result:
                raise ValueError(f"duplicate {domain} stem: {path.stem}")
            result[path.stem] = path
        return result

    low = index(low_files, "low")
    high = index(high_files, "high")
    if low.keys() != high.keys():
        raise ValueError(
            "paired validation filenames do not match: "
            f"missing high={sorted(low.keys() - high.keys())[:10]}, "
            f"missing low={sorted(high.keys() - low.keys())[:10]}"
        )
    return [InputRecord(low[stem], high[stem]) for stem in sorted(low)]


def resolve_validation_records(experiment_dir: Path, config: dict) -> list[InputRecord]:
    """Resolve the full-resolution validation order used by training."""
    data = config.get("data", {})
    mode = str(data.get("mode", "unknown"))
    dataset_root = Path(str(data.get("path", ""))).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (experiment_dir.parents[3] / dataset_root).resolve()
    split_root = dataset_root / "val"
    if not split_root.is_dir():
        split_root = dataset_root / "test"
    if not split_root.is_dir():
        raise FileNotFoundError(f"dataset has neither val nor test split: {dataset_root}")

    low_root = split_root / "low"
    if mode in {"pure_low_single", "pure_low_double"} and not low_root.is_dir():
        low_root = split_root / "images"
    low_files = _image_files(low_root)
    high_files = _image_files(split_root / "high", required=False)

    if mode == "paired":
        if not high_files:
            raise FileNotFoundError(f"paired validation lacks high images: {split_root / 'high'}")
        return _pair_by_stem(low_files, high_files)

    # A matched high image is optional diagnostic information for pure-low-single.
    # It is used only if every filename matches exactly; partial/positional pairing
    # would silently create invalid metrics and is therefore rejected by omission.
    optional_high: dict[str, Path] = {}
    if mode == "pure_low_single" and high_files:
        high_by_stem = {path.stem: path for path in high_files}
        if len(high_by_stem) == len(high_files) and all(
            path.stem in high_by_stem for path in low_files
        ):
            optional_high = high_by_stem
    return [InputRecord(path, optional_high.get(path.stem)) for path in low_files]


def _manifest_records(experiment_dir: Path, full_records: list[InputRecord]) -> list[InputRecord]:
    manifest = experiment_dir / "quick_val_manifest.txt"
    if not manifest.is_file():
        return []
    by_low = {str(record.low.resolve()): record for record in full_records}
    crop_size: int | None = None
    selected: list[InputRecord] = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("crop_size:"):
            value = int(line.split(":", 1)[1].strip())
            crop_size = value if value > 0 else None
        elif line and not ":" in line:
            key = str(Path(line).expanduser().resolve())
            if key not in by_low:
                raise ValueError(f"quick validation manifest path is not in validation set: {line}")
            base = by_low[key]
            selected.append(InputRecord(base.low, base.high, crop_size))
    return selected


def records_for_iteration(
    experiment_dir: Path,
    iteration_dir: Path,
    full_records: list[InputRecord],
) -> list[InputRecord]:
    """Choose full or quick validation records without positional guessing."""
    report_path = experiment_dir / "final_full_validation.yaml"
    if report_path.is_file():
        report = yaml.safe_load(report_path.read_text(encoding="utf-8")) or {}
        published = report.get("published_image_dir")
        if published and (experiment_dir / str(published)).resolve() == iteration_dir.resolve():
            return full_records

    quick_records = _manifest_records(experiment_dir, full_records)
    if quick_records:
        return quick_records
    return full_records


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise OSError(f"cannot read image: {path}")
    return image.astype(np.float32) / 255.0


def gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    # Saved/input images are BGR because cv2 is used consistently.
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    if size > height or size > width:
        raise ValueError(f"center crop {size} exceeds image shape {image.shape}")
    top = (height - size) // 2
    left = (width - size) // 2
    return image[top:top + size, left:left + size, ...]


def align_input(image: np.ndarray, output_shape: tuple[int, int], crop: int | None) -> np.ndarray:
    if image.shape[:2] == output_shape:
        return image
    if crop is not None:
        image = center_crop(image, crop)
    if image.shape[:2] != output_shape:
        raise ValueError(
            f"saved output shape {output_shape} does not match validation input {image.shape[:2]}"
        )
    return image


def total_variation(image: np.ndarray) -> float:
    terms = []
    if image.shape[1] > 1:
        terms.append(float(np.abs(image[:, 1:, ...] - image[:, :-1, ...]).mean()))
    if image.shape[0] > 1:
        terms.append(float(np.abs(image[1:, ...] - image[:-1, ...]).mean()))
    return sum(terms)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denominator < 1e-12 else float(np.dot(a, b) / denominator)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    minimum = min(a.shape[0], a.shape[1])
    win_size = min(7, minimum if minimum % 2 else minimum - 1)
    if win_size < 3:
        return float("nan")
    return float(structural_similarity(
        a, b, channel_axis=2 if a.ndim == 3 else None,
        data_range=1.0, win_size=win_size,
    ))


def chroma(image: np.ndarray) -> np.ndarray:
    return image / np.maximum(image.sum(axis=2, keepdims=True), 1e-4)


def comparison_metrics(
    a: np.ndarray, b: np.ndarray, prefix: str, include_ssim: bool = False,
) -> dict[str, float]:
    difference = a - b
    result = {
        f"{prefix}_l1": float(np.abs(difference).mean()),
        f"{prefix}_psnr": psnr(a, b),
    }
    if include_ssim:
        result[f"{prefix}_ssim"] = ssim(a, b)
    return result


def reflectance_metrics(reflectance: np.ndarray, prefix: str) -> dict[str, float]:
    reflectance_gray = gray(reflectance)
    return {
        f"{prefix}_mean": float(reflectance.mean()),
        f"{prefix}_std": float(reflectance.std()),
        f"{prefix}_gray_std": float(reflectance_gray.std()),
        f"{prefix}_sat_black_001": float((reflectance < 0.01).mean()),
        f"{prefix}_sat_white_099": float((reflectance > 0.99).mean()),
        f"{prefix}_dark_005": float((reflectance < 0.05).mean()),
        f"{prefix}_bright_095": float((reflectance > 0.95).mean()),
        f"{prefix}_gradient": total_variation(reflectance_gray),
    }


def illumination_metrics(illumination: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(illumination.mean()),
        f"{prefix}_std": float(illumination.std()),
        f"{prefix}_dark_005": float((illumination < 0.05).mean()),
        f"{prefix}_bright_095": float((illumination > 0.95).mean()),
        f"{prefix}_tv": total_variation(illumination),
    }


def consistency_metrics(r_low: np.ndarray, r_high: np.ndarray) -> dict[str, float]:
    if r_low.shape != r_high.shape:
        raise ValueError(f"R_low/R_high shape mismatch: {r_low.shape} vs {r_high.shape}")
    result = comparison_metrics(r_low, r_high, "r_consistency", include_ssim=True)
    result["r_consistency_chroma_l1"] = float(
        np.abs(chroma(r_low) - chroma(r_high)).mean()
    )
    return result


def indexed_paths(iteration_dir: Path, suffix: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    marker = f"_{suffix}"
    for path in iteration_dir.glob(f"*_{suffix}.png"):
        index_text = path.stem.removesuffix(marker)
        if not index_text.isdigit():
            continue
        index = int(index_text)
        if index in result:
            raise ValueError(f"duplicate image index {index}: {iteration_dir}")
        result[index] = path
    return result


def anchor_diagnostics(
    illumination: np.ndarray,
    input_low: np.ndarray,
    anchor_version: str,
    l_type: str,
) -> tuple[float, float]:
    """Return target summary and the exact per-image training anchor loss."""
    prediction = float(illumination.mean())
    if l_type == "point":
        if anchor_version == "v1":
            target_map = input_low.max(axis=2)
            return float(target_map.mean()), float(np.abs(prediction - target_map).mean())
        target = float(input_low.max())
    elif l_type == "pixel":
        target = (
            float(input_low.max(axis=2).mean())
            if anchor_version == "v1" else float(input_low.mean())
        )
    else:
        raise ValueError(f"unknown L type for anchor diagnostics: {l_type}")
    return target, abs(prediction - target)


def analyze_iteration(
    iteration_dir: Path,
    paired: bool = False,
    input_records: list[InputRecord] | None = None,
    mode: str | None = None,
    anchor_version: str = "v2",
    l_type: str = "pixel",
) -> tuple[list[dict], list[str]]:
    """Analyze one saved iteration.

    ``paired`` is retained for compatibility with existing callers.  New callers
    should also pass ``mode`` and ``input_records`` to enable reference metrics.
    """
    mode = mode or ("paired" if paired else "unknown")
    paired = paired or mode == "paired"
    warnings: list[str] = []
    paths = {
        (component, domain): indexed_paths(iteration_dir, f"{component}_{domain}")
        for component in ("R", "L") for domain in ("low", "high")
    }
    low_indices = set(paths[("R", "low")])
    missing_low_l = sorted(low_indices - set(paths[("L", "low")]))
    if missing_low_l:
        raise FileNotFoundError(f"iter {iteration_dir.name} missing L_low: {missing_low_l[:10]}")
    if not low_indices:
        return [], [f"iter {iteration_dir.name}: no *_R_low.png"]
    if low_indices != set(range(max(low_indices) + 1)):
        raise ValueError(f"iter {iteration_dir.name} has non-contiguous output indices")
    if input_records is not None and max(low_indices) >= len(input_records):
        raise ValueError(
            f"iter {iteration_dir.name} has {max(low_indices)+1} outputs but only "
            f"{len(input_records)} validation records"
        )

    high_indices = set(paths[("R", "high")])
    if paired:
        missing_high = sorted(low_indices - high_indices)
        missing_high_l = sorted(high_indices - set(paths[("L", "high")]))
        if missing_high:
            warnings.append(
                f"iter {iteration_dir.name}: paired results miss {len(missing_high)} R_high images"
            )
        if missing_high_l:
            warnings.append(
                f"iter {iteration_dir.name}: results miss {len(missing_high_l)} L_high images"
            )

    rows: list[dict] = []
    for index in sorted(low_indices):
        r_low = read_image(paths[("R", "low")][index])
        l_low = read_image(paths[("L", "low")][index], grayscale=True)
        if r_low.shape[:2] != l_low.shape:
            raise ValueError(f"R_low/L_low shape mismatch at index {index}")
        row: dict[str, int | float | str] = {
            "iteration": iteration_value(iteration_dir),
            "image_set": iteration_dir.name,
            "image_index": index,
        }
        row.update(reflectance_metrics(r_low, "r_low"))
        row.update(illumination_metrics(l_low, "l_low"))

        input_low: np.ndarray | None = None
        input_high: np.ndarray | None = None
        if input_records is not None:
            record = input_records[index]
            row["input_low_file"] = record.low.name
            input_low = align_input(read_image(record.low), r_low.shape[:2], record.center_crop)
            input_gray = gray(input_low)
            input_tv = total_variation(input_gray)
            row.update({
                "input_low_mean": float(input_low.mean()),
                "input_low_gray_std": float(input_gray.std()),
                "input_low_tv": input_tv,
                "r_low_mean_gain_vs_input": float(r_low.mean()) / max(float(input_low.mean()), 1e-8),
                "r_low_contrast_gain_vs_input": float(gray(r_low).std()) / max(float(input_gray.std()), 1e-8),
                "r_low_tv_to_input": total_variation(gray(r_low)) / max(input_tv, 1e-8),
                "r_low_input_gray_corr": pearson(gray(r_low), input_gray),
                "r_low_input_chroma_l1": float(np.abs(chroma(r_low) - chroma(input_low)).mean()),
                "l_low_tv_to_input": total_variation(l_low) / max(input_tv, 1e-8),
                "l_low_input_gray_corr": pearson(l_low, input_gray),
            })
            synthesis_low = np.clip(r_low * l_low[..., None], 0.0, 1.0)
            row.update(comparison_metrics(synthesis_low, input_low, "self_low"))
            if mode == "pure_low_single":
                target, anchor_error = anchor_diagnostics(
                    l_low, input_low, anchor_version, l_type
                )
                row["anchor_target"] = target
                row["anchor_abs_error"] = anchor_error
            if record.high is not None:
                input_high = align_input(
                    read_image(record.high), r_low.shape[:2], record.center_crop
                )
                row["input_high_file"] = record.high.name
                row["input_high_mean"] = float(input_high.mean())
                row.update(comparison_metrics(
                    input_low, input_high, "input_low_highref", include_ssim=True
                ))
                row.update(comparison_metrics(
                    r_low, input_high, "r_low_highref", include_ssim=True
                ))
                high_gray = gray(input_high)
                row.update({
                    "r_low_highref_mean_ratio": float(r_low.mean()) / max(float(input_high.mean()), 1e-8),
                    "r_low_highref_overbright_010": float((gray(r_low) > high_gray + 0.1).mean()),
                    "r_low_highref_chroma_l1": float(np.abs(chroma(r_low) - chroma(input_high)).mean()),
                    "r_low_highref_gray_corr": pearson(gray(r_low), high_gray),
                    "r_low_highref_psnr_gain_vs_input": (
                        row["r_low_highref_psnr"] - row["input_low_highref_psnr"]
                    ),
                })

        if paired and index in high_indices:
            r_high = read_image(paths[("R", "high")][index])
            row.update(reflectance_metrics(r_high, "r_high"))
            row.update(consistency_metrics(r_low, r_high))
            if index in paths[("L", "high")]:
                l_high = read_image(paths[("L", "high")][index], grayscale=True)
                if l_high.shape != r_high.shape[:2]:
                    raise ValueError(f"R_high/L_high shape mismatch at index {index}")
                row.update(illumination_metrics(l_high, "l_high"))
                if input_high is not None:
                    high_gray = gray(input_high)
                    high_tv = total_variation(high_gray)
                    row.update({
                        "l_high_tv_to_input": total_variation(l_high) / max(high_tv, 1e-8),
                        "l_high_input_gray_corr": pearson(l_high, high_gray),
                    })
                    synthesis_high = np.clip(r_high * l_high[..., None], 0.0, 1.0)
                    row.update(comparison_metrics(synthesis_high, input_high, "self_high"))
                    row.update(comparison_metrics(r_high, input_high, "r_high_highref"))
                    cross_low = np.clip(r_high * l_low[..., None], 0.0, 1.0)
                    cross_high = np.clip(r_low * l_high[..., None], 0.0, 1.0)
                    row.update(comparison_metrics(cross_low, input_low, "cross_low"))
                    row.update(comparison_metrics(cross_high, input_high, "cross_high"))
        rows.append(row)
    return rows, warnings


def _finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def summarize(rows: list[dict]) -> list[dict]:
    by_iteration: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_iteration[int(row["iteration"])].append(row)
    summaries: list[dict] = []
    for iteration, iteration_rows in sorted(by_iteration.items()):
        summary: dict[str, int | float] = {
            "iteration": iteration,
            "n_low": len(iteration_rows),
            "n_paired": sum("r_consistency_l1" in row for row in iteration_rows),
            "n_highref": sum("r_low_highref_l1" in row for row in iteration_rows),
        }
        keys = sorted(set().union(*(row.keys() for row in iteration_rows)))
        for key in keys:
            if key in {"iteration", "image_set", "image_index", "input_low_file", "input_high_file"}:
                continue
            values = _finite([float(row[key]) for row in iteration_rows if key in row])
            if not len(values):
                continue
            stats = {
                "mean": values.mean(), "std": values.std(),
                "p05": np.quantile(values, 0.05), "median": np.quantile(values, 0.5),
                "p95": np.quantile(values, 0.95), "min": values.min(), "max": values.max(),
            }
            for stat, value in stats.items():
                summary[f"{key}_{stat}"] = float(value)
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_table(columns: list[tuple[str, str, str]], summaries: list[dict]) -> list[str]:
    header = "  ".join(f"{label:>13}" for label, _, _ in columns)
    lines = [header, "  ".join("-" * 13 for _ in columns)]
    for summary in summaries:
        values = []
        for _, key, spec in columns:
            value = summary.get(key)
            values.append(f"{value:13{spec}}" if value is not None else f"{'-':>13}")
        lines.append("  ".join(values))
    return lines


def write_report(
    path: Path, experiment_dir: Path, mode: str,
    summaries: list[dict], warnings: list[str], source_signature: str | None = None,
) -> None:
    lines = [
        f"# Retinex decomposition analysis v{ANALYSIS_VERSION}: {experiment_dir.name}",
        f"mode: {mode}", "",
        "All metrics use saved 8-bit PNG components and the validation order from config.yaml.",
        "Self reconstruction measures R*L vs input; it does not prove decomposition semantics.",
    ]
    if source_signature is not None:
        lines.insert(2, f"source_signature: {source_signature}")
    if mode == "paired":
        lines.append(
            "R consistency measures R_low vs R_high; two jointly wrong R images can still score well."
        )
    if any("r_low_highref_psnr_mean" in summary for summary in summaries):
        lines.extend([
            "High-reference metrics use a matched normal-light photograph as a diagnostic reference,",
            "not as literal reflectance ground truth.",
        ])
    lines.append("")
    if warnings:
        lines.extend(["Warnings:", *(f"- {warning}" for warning in warnings), ""])

    common = [
        ("iter", "iteration", ".0f"), ("n", "n_low", ".0f"),
        ("Rlow_mean", "r_low_mean_mean", ".4f"),
        ("Rmean/input", "r_low_mean_gain_vs_input_mean", ".3f"),
        ("Rlow_std", "r_low_std_mean", ".4f"),
        ("Rlow_>0.95", "r_low_bright_095_mean", ".4f"),
        ("R_TV/input", "r_low_tv_to_input_mean", ".3f"),
        ("corr(R,I)", "r_low_input_gray_corr_mean", ".3f"),
        ("Llow_mean", "l_low_mean_mean", ".4f"),
        ("L_TV/input", "l_low_tv_to_input_mean", ".3f"),
        ("corr(L,I)", "l_low_input_gray_corr_mean", ".3f"),
    ]
    if mode == "pure_low_single":
        common.append(("anchor_err", "anchor_abs_error_mean", ".4f"))
    lines.extend(["Component and no-reference structure diagnostics:", *_format_table(common, summaries), ""])

    reconstruction = [
        ("iter", "iteration", ".0f"), ("n", "n_low", ".0f"),
        ("selfL_L1", "self_low_l1_mean", ".5f"),
        ("selfL_PSNR", "self_low_psnr_mean", ".2f"),
    ]
    if mode == "paired":
        reconstruction.extend([
            ("selfH_L1", "self_high_l1_mean", ".5f"),
            ("selfH_PSNR", "self_high_psnr_mean", ".2f"),
            ("crossL_PSNR", "cross_low_psnr_mean", ".2f"),
            ("crossH_PSNR", "cross_high_psnr_mean", ".2f"),
        ])
    lines.extend([
        "Reconstruction-integrity diagnostics (not decomposition quality):",
        *_format_table(reconstruction, summaries), "",
    ])

    if any("r_low_highref_psnr_mean" in summary for summary in summaries):
        reference = [
            ("iter", "iteration", ".0f"), ("n_ref", "n_highref", ".0f"),
            ("low-ref_PSNR", "input_low_highref_psnr_mean", ".2f"),
            ("Rlow-ref_L1", "r_low_highref_l1_mean", ".4f"),
            ("Rlow-ref_PSNR", "r_low_highref_psnr_mean", ".2f"),
            ("Rlow-ref_SSIM", "r_low_highref_ssim_mean", ".3f"),
            ("mean_ratio", "r_low_highref_mean_ratio_mean", ".3f"),
            (">ref+0.1", "r_low_highref_overbright_010_mean", ".3f"),
            ("chroma_L1", "r_low_highref_chroma_l1_mean", ".4f"),
            ("gray_corr", "r_low_highref_gray_corr_mean", ".3f"),
            ("PSNR_gain", "r_low_highref_psnr_gain_vs_input_mean", ".2f"),
        ]
        lines.extend(["Matched normal-light reference diagnostics:", *_format_table(reference, summaries), ""])

    if mode == "paired":
        paired_columns = [
            ("iter", "iteration", ".0f"), ("n_pair", "n_paired", ".0f"),
            ("R_LH_L1", "r_consistency_l1_mean", ".4f"),
            ("R_LH_PSNR", "r_consistency_psnr_mean", ".2f"),
            ("R_LH_SSIM", "r_consistency_ssim_mean", ".3f"),
            ("Rhigh-ref", "r_high_highref_psnr_mean", ".2f"),
            ("Lhigh_mean", "l_high_mean_mean", ".4f"),
            ("Lhigh_>0.95", "l_high_bright_095_mean", ".3f"),
            ("Lh_TV/input", "l_high_tv_to_input_mean", ".3f"),
            ("corr(Lh,Ih)", "l_high_input_gray_corr_mean", ".3f"),
        ]
        lines.extend(["Paired-only consistency and high-domain diagnostics:", *_format_table(paired_columns, summaries), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def sources_signature(sources: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in sources}):
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def is_analysis_fresh(report: Path, sources: list[Path]) -> bool:
    if not report.is_file():
        return False
    expected = f"source_signature: {sources_signature(sources)}"
    return expected in report.read_text(encoding="utf-8").splitlines()[:5]


def main() -> int:
    args = parse_args()
    try:
        experiment_dir = resolve_experiment(args.experiment, args.experiments_dir)
        config = load_config(experiment_dir)
        mode = str(config.get("data", {}).get("mode", "unknown"))
        loss = config.get("loss", {})
        anchor_version = str(loss.get("anchor_version", "v2"))
        loss_mode = str(loss.get("mode", ""))
        l_type = "point" if loss_mode.endswith("_point") else "pixel"
        iteration_dirs, init_warnings = find_iteration_dirs(
            experiment_dir,
            args.iteration,
            selected_image_sets=args.image_set,
            all_image_sets=args.all_image_sets,
        )
        full_records = resolve_validation_records(experiment_dir, config)
        report_path = experiment_dir / f"{args.output_prefix}.txt"

        source_paths = [Path(__file__), experiment_dir / "config.yaml"]
        source_paths.extend(
            path for path in (
                experiment_dir / "quick_val_manifest.txt",
                experiment_dir / "final_full_validation.yaml",
            ) if path.is_file()
        )
        for directory in iteration_dirs:
            source_paths.extend(directory.glob("*.png"))
            metadata_path = directory / "_image_set.yaml"
            if metadata_path.is_file():
                source_paths.append(metadata_path)
        source_paths.extend(record.low for record in full_records)
        source_paths.extend(record.high for record in full_records if record.high is not None)
        details_path = experiment_dir / f"{args.output_prefix}_details.csv"
        details_ready = not args.details or details_path.is_file()
        if not args.force and details_ready and is_analysis_fresh(report_path, source_paths):
            print(f"[SKIP] {experiment_dir} — analysis is fresh")
            return 0

        detail_rows: list[dict] = []
        warnings: list[str] = list(init_warnings)
        for iteration_dir in iteration_dirs:
            records = records_for_iteration(experiment_dir, iteration_dir, full_records)
            rows, iteration_warnings = analyze_iteration(
                iteration_dir, paired=mode == "paired", input_records=records,
                mode=mode, anchor_version=anchor_version, l_type=l_type,
            )
            detail_rows.extend(rows)
            warnings.extend(iteration_warnings)
        summaries = summarize(detail_rows)
        write_report(
            report_path, experiment_dir, mode, summaries, warnings,
            source_signature=sources_signature(source_paths),
        )
        output_paths = [report_path]
        if args.details:
            write_csv(details_path, detail_rows)
            output_paths.append(details_path)
        print(f"experiment: {experiment_dir}")
        print(f"mode: {mode}; iterations: {[row['iteration'] for row in summaries]}")
        for output in output_paths:
            print(f"saved: {output}")
        for warning in warnings:
            print(f"[WARN] {warning}", file=sys.stderr)
        return 0
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
