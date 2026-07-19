#!/usr/bin/env python3
"""Shared utilities for RetinexPixelTrans paired analysis steps."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[4]
EXP_ROOT = ROOT / "experiments" / "RetinexPixelTrans" / "paired"
STEP_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = STEP_ROOT / "results"
FIG_ROOT = RESULT_ROOT / "figures"
COMPARE_ANALYZER = ROOT / "_compare" / "analyze_decomposition.py"
DEFAULT_IMAGE_SET = "best"

FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def ensure_output_dirs() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)


def discover_runs() -> list[Path]:
    if not EXP_ROOT.is_dir():
        return []
    runs = []
    for run_dir in sorted(EXP_ROOT.iterdir(), key=lambda path: path.name):
        if not run_dir.is_dir():
            continue
        if (run_dir / "config.yaml").is_file():
            runs.append(run_dir)
    return runs


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_config(run_dir: Path) -> dict[str, Any]:
    return load_yaml(run_dir / "config.yaml")


def nested_get(data: dict[str, Any], dotted: str, default: Any = "") -> Any:
    current: Any = data
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def config_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    loss = config.get("loss", {}) or {}
    training = config.get("training", {}) or {}
    data = config.get("data", {}) or {}
    model = config.get("model", {}) or {}
    return {
        "model.name": model.get("name", ""),
        "model.dim": model.get("dim", ""),
        "data.mode": data.get("mode", ""),
        "data.path": data.get("path", ""),
        "data.batch_size": data.get("batch_size", ""),
        "data.crop_size": data.get("crop_size", ""),
        "training.seed": training.get("seed", ""),
        "training.max_iterations": training.get("max_iterations", ""),
        "loss.mode": loss.get("mode", ""),
        "loss.recon_weight_high": loss.get("recon_weight_high", ""),
        "loss.recon_weight_low": loss.get("recon_weight_low", ""),
        "loss.cross_recon_weight_high": loss.get("cross_recon_weight_high", ""),
        "loss.cross_recon_weight_low": loss.get("cross_recon_weight_low", ""),
        "loss.equal_r_weight": loss.get("equal_r_weight", ""),
        "loss.smooth_weight": loss.get("smooth_weight", ""),
        "loss.smooth_version": loss.get("smooth_version", ""),
    }


def run_label(run_dir: Path, config: dict[str, Any] | None = None) -> str:
    config = config if config is not None else run_config(run_dir)
    loss = config.get("loss", {}) or {}
    cross = loss.get("cross_recon_weight_high", "")
    equal = loss.get("equal_r_weight", "")
    smooth = loss.get("smooth_weight", "")
    version = loss.get("smooth_version", "") or infer_smooth_version(run_dir.name)
    return f"cr={cross} er={equal} sm={smooth}{version}"


def infer_smooth_version(run_name: str) -> str:
    match = re.search(r"smv(\d+)", run_name)
    return f"v{match.group(1)}" if match else "v1"


def add_image_set_args(parser) -> None:
    parser.add_argument("--image-set", default=DEFAULT_IMAGE_SET)
    parser.add_argument("--iteration", type=int, default=None)


def is_selectable_image_set(name: str) -> bool:
    return name in {DEFAULT_IMAGE_SET, "final_best"} or name.isdigit()


def selected_image_set(args) -> str:
    return str(args.iteration) if args.iteration is not None else str(args.image_set)


def analyzer_args_for_image_set(args) -> list[str]:
    if args.iteration is not None:
        return ["--iteration", str(args.iteration)]
    return ["--image-set", str(args.image_set)]


def detail_rows_for_image_set(rows: list[dict[str, str]], args) -> list[dict[str, str]]:
    if args.iteration is not None:
        return [row for row in rows if int(float(row.get("iteration", -1))) == args.iteration]
    image_set = str(args.image_set)
    if rows and "image_set" in rows[0]:
        return [row for row in rows if row.get("image_set") == image_set]
    if image_set.isdigit():
        return [row for row in rows if int(float(row.get("iteration", -1))) == int(image_set)]
    return []


def iteration_dirs(run_dir: Path) -> list[str]:
    img_root = run_dir / "img"
    if not img_root.is_dir():
        return []
    return sorted(path.name for path in img_root.iterdir() if path.is_dir() and is_selectable_image_set(path.name))


def component_counts(run_dir: Path, image_set: str | int) -> dict[str, int]:
    image_dir = run_dir / "img" / str(image_set)
    counts: dict[str, int] = {}
    for suffix in ("R_low", "L_low", "R_high", "L_high"):
        counts[suffix] = len(list(image_dir.glob(f"*_{suffix}.png"))) if image_dir.is_dir() else 0
    return counts


def synthesis_dir_for_image_set(run_dir: Path, image_set: str | int) -> Path:
    image_set = str(image_set)
    direct = run_dir / "synthesis" / image_set
    if direct.is_dir():
        return direct
    metadata_path = run_dir / "img" / image_set / "_image_set.yaml"
    if metadata_path.is_file():
        metadata = load_yaml(metadata_path)
        checkpoint_step = metadata.get("checkpoint_step")
        if checkpoint_step is not None:
            fallback = run_dir / "synthesis" / str(checkpoint_step)
            if fallback.is_dir():
                return fallback
    return direct


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float]) -> float:
    valid = [value for value in values if math.isfinite(value)]
    return sum(valid) / len(valid) if valid else math.nan


def quantile(values: Iterable[float], q: float) -> float:
    valid = sorted(value for value in values if math.isfinite(value))
    if not valid:
        return math.nan
    if len(valid) == 1:
        return valid[0]
    position = (len(valid) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return valid[low]
    weight = position - low
    return valid[low] * (1 - weight) + valid[high] * weight


def summarize_metric(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [as_float(row, key) for row in rows]
    return {
        f"{key}_mean": mean(values),
        f"{key}_median": quantile(values, 0.50),
        f"{key}_p05": quantile(values, 0.05),
        f"{key}_p95": quantile(values, 0.95),
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for label, _, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for _, key, fmt in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append("-" if not math.isfinite(value) else format(value, fmt))
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def details_path(run_dir: Path, prefix: str = "decomposition_analysis") -> Path:
    return run_dir / f"{prefix}_details.csv"


def report_path(run_dir: Path, prefix: str = "decomposition_analysis") -> Path:
    return run_dir / f"{prefix}.txt"


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
