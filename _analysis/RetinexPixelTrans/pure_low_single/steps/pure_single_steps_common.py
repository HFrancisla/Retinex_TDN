#!/usr/bin/env python3
"""Shared helpers for RetinexPixelTrans pure-low-single analysis steps."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[4]
EXP_ROOT = ROOT / "experiments" / "RetinexPixelTrans" / "pure_low_single"
STEP_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = STEP_ROOT / "results"
FIG_ROOT = RESULT_ROOT / "figures"
COMPARE_ANALYZER = ROOT / "_compare" / "analyze_decomposition.py"


def ensure_output_dirs() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)


def discover_runs() -> list[Path]:
    if not EXP_ROOT.is_dir():
        return []
    return sorted(
        [path for path in EXP_ROOT.iterdir() if path.is_dir() and (path / "config.yaml").is_file()],
        key=lambda path: path.name,
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_config(run_dir: Path) -> dict[str, Any]:
    return load_yaml(run_dir / "config.yaml")


def infer_dataset(config: dict[str, Any], run_name: str = "") -> str:
    data_path = str((config.get("data", {}) or {}).get("path", ""))
    text = f"{run_name} {data_path}".lower()
    if "bddnight" in text:
        return "BDDnight"
    if "lolv2" in text:
        return "LOLv2"
    return Path(data_path).name or "unknown"


def infer_smooth_version(run_name: str, config: dict[str, Any]) -> str:
    loss = config.get("loss", {}) or {}
    if loss.get("smooth_version"):
        return str(loss["smooth_version"])
    match = re.search(r"smv(\d+)", run_name)
    return f"v{match.group(1)}" if match else "v1"


def config_fingerprint(config: dict[str, Any], run_name: str = "") -> dict[str, Any]:
    loss = config.get("loss", {}) or {}
    training = config.get("training", {}) or {}
    data = config.get("data", {}) or {}
    model = config.get("model", {}) or {}
    return {
        "dataset": infer_dataset(config, run_name),
        "model.name": model.get("name", ""),
        "model.dim": model.get("dim", ""),
        "data.mode": data.get("mode", ""),
        "data.path": data.get("path", ""),
        "data.batch_size": data.get("batch_size", ""),
        "data.crop_size": data.get("crop_size", ""),
        "training.seed": training.get("seed", ""),
        "training.max_iterations": training.get("max_iterations", ""),
        "loss.mode": loss.get("mode", ""),
        "loss.recon_weight": loss.get("recon_weight", ""),
        "loss.anchor_version": loss.get("anchor_version", ""),
        "loss.anchor_weight": loss.get("anchor_weight", ""),
        "loss.bdsp_weight": loss.get("bdsp_weight", ""),
        "loss.smooth_weight": loss.get("smooth_weight", ""),
        "loss.smooth_version": infer_smooth_version(run_name, config),
    }


def run_label(run_dir: Path, config: dict[str, Any] | None = None) -> str:
    config = config if config is not None else run_config(run_dir)
    loss = config.get("loss", {}) or {}
    recon_value = loss.get("recon_weight", "")
    try:
        recon_float = float(recon_value)
        recon = str(int(recon_float)) if recon_float.is_integer() else str(recon_float)
    except (TypeError, ValueError):
        recon = str(recon_value)
    anchor = str(loss.get("anchor_version", "")).replace("v", "")
    smooth_value = loss.get("smooth_weight", "")
    try:
        smooth_float = float(smooth_value)
        smooth = str(int(smooth_float)) if smooth_float.is_integer() else str(smooth_float)
    except (TypeError, ValueError):
        smooth = str(smooth_value)
    smooth_version = infer_smooth_version(run_dir.name, config)
    return f"r{recon}-a{anchor}-sm{smooth}{smooth_version}"


def iteration_dirs(run_dir: Path) -> list[int]:
    img_root = run_dir / "img"
    if not img_root.is_dir():
        return []
    return sorted(int(path.name) for path in img_root.iterdir() if path.is_dir() and path.name.isdigit())


def component_counts(run_dir: Path, iteration: int) -> dict[str, int]:
    image_dir = run_dir / "img" / str(iteration)
    return {
        suffix: len(list(image_dir.glob(f"*_{suffix}.png"))) if image_dir.is_dir() else 0
        for suffix in ("R_low", "L_low")
    }


def details_path(run_dir: Path, prefix: str = "decomposition_analysis") -> Path:
    return run_dir / f"{prefix}_details.csv"


def report_path(run_dir: Path, prefix: str = "decomposition_analysis") -> Path:
    return run_dir / f"{prefix}.txt"


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
    pos = (len(valid) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return valid[low]
    weight = pos - low
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


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def dataset_input_roots(config: dict[str, Any], run_name: str = "") -> tuple[Path, Path | None]:
    data_root = Path(str((config.get("data", {}) or {}).get("path", ""))).expanduser()
    if not data_root.is_absolute():
        data_root = (ROOT / data_root).resolve()
    dataset = infer_dataset(config, run_name)
    if dataset == "BDDnight":
        return data_root / "val" / "images", None
    split = data_root / "val"
    if not split.is_dir():
        split = data_root / "test"
    return split / "low", split / "high"


def full_validation_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "final_full_validation.yaml"
    if not path.is_file():
        return {}
    data = load_yaml(path)
    result: dict[str, Any] = {
        "full_sample_count": data.get("sample_count", ""),
        "full_selection_metric": data.get("selection_metric", ""),
        "full_selected_checkpoint_step": data.get("selected_checkpoint_step", ""),
        "full_selected_value": data.get("selected_full_value", ""),
    }
    selected = None
    for candidate in data.get("candidates", []) or []:
        if candidate.get("selected"):
            selected = candidate
            break
    if selected:
        for key, value in (selected.get("metrics", {}) or {}).items():
            result[f"full_{key}"] = value
    return result
