#!/usr/bin/env python3
"""
Retinex R/L 分解结果诊断工具。

用途：
    分析指定实验 ``img/<iteration>/`` 中保存的验证集分解图，统计 R 的
    均值、标准差、饱和率和梯度强度，以及 L 的 total variation。对于
    ``data.mode: paired`` 实验，如果同时保存了 high 分解，还会计算同场景
    R_low/R_high 的 L1 和 PSNR 一致性；其他训练模式只分析 low。

输入文件命名：
    <index>_R_low.png、<index>_L_low.png
    paired 可额外包含 <index>_R_high.png、<index>_L_high.png

输出：
    默认在实验根目录生成 ``decomposition_analysis.txt``。添加 ``--details``
    时额外生成 ``decomposition_analysis_details.csv``，用于定位单张异常图。

用法：
    # 使用完整实验路径，分析 img 下全部迭代
    .venv/bin/python _compare/analyze_decomposition.py \
        experiments/RetinexPixelTrans/paired/<run>

    # 使用指定实验根目录，只分析 iteration 10000
    .venv/bin/python _compare/analyze_decomposition.py <run-name> \
        --experiments-dir experiments/RetinexPixelTrans/paired \
        --iteration 10000

    # 额外保存逐图片指标
    .venv/bin/python _compare/analyze_decomposition.py <实验目录> \
        --iteration 10000 --details
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_PREFIX = "decomposition_analysis"
DOMAINS = ("low", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        help="实验 run 目录，或相对于 --experiments-dir 的目录名",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="experiment 使用相对名称时的根目录（默认：experiments）",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        action="append",
        help="只分析指定迭代；可重复传入。默认分析 img 下全部数字目录",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_PREFIX,
        help=f"输出文件名前缀（默认：{DEFAULT_PREFIX}）",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="额外输出逐图片指标 CSV；默认只生成文本汇总",
    )
    return parser.parse_args()


def resolve_experiment(value: str, experiments_dir: Path) -> Path:
    direct = Path(value).expanduser()
    candidates = (direct, experiments_dir.expanduser() / value)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"找不到实验目录，已尝试：{attempted}")


def load_config(experiment_dir: Path) -> dict:
    config_path = experiment_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"实验缺少 config.yaml：{config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def find_iteration_dirs(experiment_dir: Path, selected: list[int] | None) -> list[Path]:
    image_root = experiment_dir / "img"
    if not image_root.is_dir():
        raise FileNotFoundError(f"实验缺少 img 目录：{image_root}")

    directories = {
        int(path.name): path
        for path in image_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    }
    if selected:
        missing = sorted(set(selected) - directories.keys())
        if missing:
            raise FileNotFoundError(f"找不到迭代目录：{missing}")
        return [directories[value] for value in sorted(set(selected))]
    if not directories:
        raise FileNotFoundError(f"img 下没有数字迭代目录：{image_root}")
    return [directories[value] for value in sorted(directories)]


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise OSError(f"无法读取图像：{path}")
    return image.astype(np.float32) / 255.0


def total_variation(image: np.ndarray) -> float:
    """相邻像素绝对差的横向均值与纵向均值之和。"""
    terms = []
    if image.shape[1] > 1:
        terms.append(float(np.abs(image[:, 1:, ...] - image[:, :-1, ...]).mean()))
    if image.shape[0] > 1:
        terms.append(float(np.abs(image[1:, ...] - image[:-1, ...]).mean()))
    return sum(terms)


def reflectance_metrics(reflectance: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(reflectance.mean()),
        f"{prefix}_std": float(reflectance.std()),
        f"{prefix}_sat_black_001": float((reflectance < 0.01).mean()),
        f"{prefix}_sat_white_099": float((reflectance > 0.99).mean()),
        f"{prefix}_dark_005": float((reflectance < 0.05).mean()),
        f"{prefix}_bright_095": float((reflectance > 0.95).mean()),
        f"{prefix}_gradient": total_variation(reflectance),
    }


def illumination_metrics(illumination: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(illumination.mean()),
        f"{prefix}_std": float(illumination.std()),
        f"{prefix}_tv": total_variation(illumination),
    }


def consistency_metrics(r_low: np.ndarray, r_high: np.ndarray) -> dict[str, float]:
    if r_low.shape != r_high.shape:
        raise ValueError(
            f"R_low/R_high 尺寸不同：low={r_low.shape}, high={r_high.shape}"
        )
    difference = r_low - r_high
    mse = float(np.mean(difference * difference))
    return {
        "r_consistency_l1": float(np.abs(difference).mean()),
        "r_consistency_psnr": 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse),
    }


def indexed_paths(iteration_dir: Path, suffix: str) -> dict[int, Path]:
    result = {}
    marker = f"_{suffix}"
    for path in iteration_dir.glob(f"*_{suffix}.png"):
        index_text = path.stem.removesuffix(marker)
        if not index_text.isdigit():
            continue
        index = int(index_text)
        if index in result:
            raise ValueError(f"重复的图像序号 {index}：{iteration_dir}")
        result[index] = path
    return result


def analyze_iteration(iteration_dir: Path, paired: bool) -> tuple[list[dict], list[str]]:
    warnings = []
    paths = {
        (component, domain): indexed_paths(iteration_dir, f"{component}_{domain}")
        for component in ("R", "L")
        for domain in DOMAINS
    }

    low_indices = set(paths[("R", "low")])
    missing_low_l = sorted(low_indices - set(paths[("L", "low")]))
    if missing_low_l:
        raise FileNotFoundError(
            f"iter {iteration_dir.name} 缺少 L_low，序号：{missing_low_l[:10]}"
        )
    if not low_indices:
        raise FileNotFoundError(f"iter {iteration_dir.name} 没有 *_R_low.png")

    high_indices = set(paths[("R", "high")])
    if paired:
        missing_high = sorted(low_indices - high_indices)
        missing_high_l = sorted(high_indices - set(paths[("L", "high")]))
        if missing_high:
            warnings.append(
                f"iter {iteration_dir.name}: paired 结果缺少 {len(missing_high)} 张 R_high；"
                "这些样本不计算 low/high 一致性"
            )
        if missing_high_l:
            warnings.append(
                f"iter {iteration_dir.name}: 缺少 {len(missing_high_l)} 张 L_high"
            )

    rows = []
    for index in sorted(low_indices):
        row: dict[str, int | float] = {
            "iteration": int(iteration_dir.name),
            "image_index": index,
        }
        r_low = read_image(paths[("R", "low")][index])
        l_low = read_image(paths[("L", "low")][index], grayscale=True)
        row.update(reflectance_metrics(r_low, "r_low"))
        row.update(illumination_metrics(l_low, "l_low"))

        # high 只有 paired 才具有同场景语义；其他模式即使遗留同名文件也忽略。
        if paired and index in high_indices:
            r_high = read_image(paths[("R", "high")][index])
            row.update(reflectance_metrics(r_high, "r_high"))
            row.update(consistency_metrics(r_low, r_high))
            if index in paths[("L", "high")]:
                l_high = read_image(paths[("L", "high")][index], grayscale=True)
                row.update(illumination_metrics(l_high, "l_high"))
        rows.append(row)
    return rows, warnings


def summarize(rows: list[dict]) -> list[dict]:
    by_iteration: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_iteration[int(row["iteration"])].append(row)

    summaries = []
    for iteration, iteration_rows in sorted(by_iteration.items()):
        summary: dict[str, int | float] = {
            "iteration": iteration,
            "n_low": len(iteration_rows),
            "n_paired": sum("r_consistency_l1" in row for row in iteration_rows),
        }
        keys = sorted(
            set().union(*(row.keys() for row in iteration_rows))
            - {"iteration", "image_index"}
        )
        for key in keys:
            values = [float(row[key]) for row in iteration_rows if key in row]
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    experiment_dir: Path,
    mode: str,
    summaries: list[dict],
    warnings: list[str],
) -> None:
    lines = [
        f"# Retinex decomposition analysis: {experiment_dir.name}",
        f"mode: {mode}",
        "",
        "指标均基于保存的 8-bit PNG；饱和率为像素/通道比例。",
        "R gradient 与 L TV 均为横、纵相邻像素绝对差均值之和。",
        "",
    ]
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    columns = [
        ("iter", "iteration", ".0f"),
        ("n", "n_low", ".0f"),
        ("paired", "n_paired", ".0f"),
        ("Rlow_mean", "r_low_mean_mean", ".5f"),
        ("Rlow_std", "r_low_std_mean", ".5f"),
        ("Rlow_>0.95", "r_low_bright_095_mean", ".5f"),
        ("Rlow_grad", "r_low_gradient_mean", ".5f"),
        ("Llow_TV", "l_low_tv_mean", ".5f"),
        ("R_LH_L1", "r_consistency_l1_mean", ".5f"),
        ("R_LH_PSNR", "r_consistency_psnr_mean", ".2f"),
    ]
    header = "  ".join(f"{label:>12}" for label, _, _ in columns)
    lines.extend([header, "  ".join("-" * 12 for _ in columns)])
    for summary in summaries:
        values = []
        for _, key, spec in columns:
            value = summary.get(key)
            values.append(f"{value:12{spec}}" if value is not None else f"{'-':>12}")
        lines.append("  ".join(values))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        experiment_dir = resolve_experiment(args.experiment, args.experiments_dir)
        config = load_config(experiment_dir)
        mode = str(config.get("data", {}).get("mode", "unknown"))
        paired = mode == "paired"
        iteration_dirs = find_iteration_dirs(experiment_dir, args.iteration)

        detail_rows = []
        warnings = []
        for iteration_dir in iteration_dirs:
            rows, iteration_warnings = analyze_iteration(iteration_dir, paired)
            detail_rows.extend(rows)
            warnings.extend(iteration_warnings)
        summaries = summarize(detail_rows)

        prefix = args.output_prefix
        report_path = experiment_dir / f"{prefix}.txt"
        write_report(report_path, experiment_dir, mode, summaries, warnings)
        output_paths = [report_path]
        if args.details:
            details_path = experiment_dir / f"{prefix}_details.csv"
            write_csv(details_path, detail_rows)
            output_paths.append(details_path)

        print(f"实验：{experiment_dir}")
        print(f"模式：{mode}；迭代：{[row['iteration'] for row in summaries]}")
        for path in output_paths:
            print(f"已保存：{path}")
        for warning in warnings:
            print(f"[WARN] {warning}", file=sys.stderr)
        return 0
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
