#!/usr/bin/env python3
"""Retinex 合成: S = R × L（逐像素相乘）。遍历 experiments/ 下 val 图像，验证分解-重建一致性。

R: 3 通道彩色反射图 [0,1]。L: 单通道光照图 [0,1]（保存时复制为 3 通道）。
    RetinexPointRaw → L 是标量，全图统一亮度系数。
    RetinexPixel*   → L 是逐像素光照图，暗处小值压暗、亮处大值提亮。

用法:
    .venv/bin/python _compare/synthesize_retinex.py                         # 全量
    RETINEX_SYNTH_MAX_ITER=10000 .venv/bin/python _compare/synthesize_retinex.py

环境变量:
    RETINEX_SYNTH_MAX_ITER  最大迭代轮次 (默认 100000)
    RETINEX_SYNTH_EXP_DIR   实验根目录 (默认 ../experiments)
    RETINEX_SYNTH_FORCE     设为 1 时强制重建

输出: img/ 同级 → synthesis/{image_set}/；仅跳过比对应 R/L 更新的有效文件。
"""

import os
import shutil
import sys
import cv2
import numpy as np
from pathlib import Path

# ============================================================
# 环境变量配置（脚本开头可修改）
# ============================================================
MAX_ITER: int = int(os.environ.get("RETINEX_SYNTH_MAX_ITER", "100000"))
"""最大处理的训练轮次。仅限制数字 image set；best/final_best 始终处理。"""

FORCE: bool = os.environ.get("RETINEX_SYNTH_FORCE", "0") == "1"
"""为 true 时无条件重建已有 synthesis 文件。"""

EXPERIMENTS_DIR: str = os.environ.get(
    "RETINEX_SYNTH_EXP_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"),
)
"""实验根目录路径。"""
# ============================================================

PREFERRED_IMAGE_SETS = ("final_best", "best")


def is_selectable_image_set(name: str) -> bool:
    if name in PREFERRED_IMAGE_SETS:
        return True
    return name.isdigit() and int(name) <= MAX_ITER


def image_set_sort_key(path: Path) -> tuple[int, int, int, str]:
    name = path.name
    if name in PREFERRED_IMAGE_SETS:
        return (0, PREFERRED_IMAGE_SETS.index(name), 0, "")
    if name.isdigit():
        return (1, 0, int(name), "")
    return (2, 0, 0, name)


def synthesize_and_save(r_path: Path, l_path: Path, out_path: Path) -> bool:
    """
    Retinex 合成: S = R × L，保存为 PNG。

    Args:
        r_path: R 分量 (3-ch BGR, uint8, [0,255])
        l_path: L 分量（按单通道读取，uint8, [0,255]）
        out_path: 输出路径

    Returns:
        True 如果实际合成了文件，False 如果已有文件仍然有效
    """
    # Existing output is valid only while it is at least as new as both inputs.
    # Revalidation may replace R/L in-place; a mere existence/count check would
    # then silently mix a new decomposition with an old synthesis.
    if (
        not FORCE
        and out_path.exists()
        and out_path.stat().st_mtime >= max(r_path.stat().st_mtime, l_path.stat().st_mtime)
    ):
        return False

    r_raw = cv2.imread(str(r_path), cv2.IMREAD_COLOR)
    l_raw = cv2.imread(str(l_path), cv2.IMREAD_GRAYSCALE)
    if r_raw is None or l_raw is None:
        raise OSError(f"无法读取 R/L: {r_path}, {l_path}")
    if r_raw.shape[:2] != l_raw.shape:
        raise ValueError(f"R/L 尺寸不一致: {r_path.name}, {l_path.name}")
    R = r_raw.astype(np.float32) / 255.0
    L = l_raw.astype(np.float32) / 255.0

    S = np.clip(R * L[..., None], 0.0, 1.0)
    S_uint8 = np.rint(S * 255.0).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    if not cv2.imwrite(str(temp_path), S_uint8):
        raise OSError(f"无法保存合成图: {temp_path}")
    os.replace(temp_path, out_path)
    return True


def process_iteration(iter_dir: Path, out_iter_dir: Path) -> dict:
    """
    处理单个迭代目录：遍历所有 R 文件，配对 L 文件，合成 S = R × L。

    Returns:
        {"low": int, "high": int}  — 本次新合成的文件数（不含已存在的）
    """
    stats = {"low": 0, "high": 0}
    expected_outputs: set[Path] = set()

    # ---- _low 分量 ----
    r_files_low = sorted(iter_dir.glob("*_R_low.png"))
    for r_file in r_files_low:
        idx = r_file.stem.replace("_R_low", "")
        l_file = iter_dir / f"{idx}_L_low.png"
        s_file = out_iter_dir / f"{idx}_S_low.png"

        if not l_file.exists():
            print(f"    [WARN] 缺少 L 文件: {l_file.name}")
            continue

        expected_outputs.add(s_file)
        if synthesize_and_save(r_file, l_file, s_file):
            stats["low"] += 1

    # ---- _high 分量（double 模式） ----
    r_files_high = sorted(iter_dir.glob("*_R_high.png"))
    for r_file in r_files_high:
        idx = r_file.stem.replace("_R_high", "")
        l_file = iter_dir / f"{idx}_L_high.png"
        s_file = out_iter_dir / f"{idx}_S_high.png"

        if not l_file.exists():
            print(f"    [WARN] 缺少 L 文件: {l_file.name}")
            continue

        expected_outputs.add(s_file)
        if synthesize_and_save(r_file, l_file, s_file):
            stats["high"] += 1

    # Derived files whose source components no longer exist must not leak into
    # PSNR reports after a revalidation or partial rerender.
    if out_iter_dir.is_dir():
        for stale in out_iter_dir.glob("*_S_*.png"):
            if stale not in expected_outputs:
                stale.unlink()
                print(f"    [REMOVE] 孤立合成文件: {stale.name}")

    return stats


def process_experiment_run(run_dir: Path) -> None:
    """处理单个实验 run 目录下所有可选 image set。"""
    img_dir = run_dir / "img"
    synthesis_dir = run_dir / "synthesis"

    if not img_dir.exists():
        print("  [SKIP] img 目录不存在")
        return

    iter_dirs = sorted(
        [d for d in img_dir.iterdir() if d.is_dir() and is_selectable_image_set(d.name)],
        key=image_set_sort_key,
    )

    if not iter_dirs:
        print(f"  [SKIP] 无可用 image set（best/final_best 或数字目录 ≤ {MAX_ITER}）")
        return

    valid_image_sets = {path.name for path in iter_dirs}
    if synthesis_dir.is_dir():
        for stale_dir in sorted(synthesis_dir.iterdir(), key=image_set_sort_key):
            if (
                stale_dir.is_dir()
                and is_selectable_image_set(stale_dir.name)
                and stale_dir.name not in valid_image_sets
            ):
                shutil.rmtree(stale_dir)
                print(f"  [REMOVE] 孤立 synthesis 目录: {stale_dir.name}")

    total_low, total_high = 0, 0

    for iter_dir in iter_dirs:
        out_iter_dir = synthesis_dir / iter_dir.name

        stats = process_iteration(iter_dir, out_iter_dir)
        n_low, n_high = stats["low"], stats["high"]
        complete_low = len(list(out_iter_dir.glob("*_S_low.png")))
        complete_high = len(list(out_iter_dir.glob("*_S_high.png")))

        parts = [f"low={complete_low} (更新 {n_low})"]
        if list(iter_dir.glob("*_R_high.png")):
            parts.append(f"high={complete_high} (更新 {n_high})")
        print(f"  [image set {iter_dir.name}] " + ", ".join(parts))

        total_low += complete_low
        total_high += complete_high

    print(f"  => 合计: {total_low} low + {total_high} high")


def main() -> None:
    exp_dir = Path(EXPERIMENTS_DIR).resolve()
    if not exp_dir.exists():
        print(f"[ERROR] 实验目录不存在: {exp_dir}")
        sys.exit(1)

    # 收集 run 目录（含 img/ 子目录的三级目录: model/mode/run）
    run_dirs: list[Path] = []
    for model_dir in sorted(exp_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for mode_dir in sorted(model_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            for run_dir in sorted(mode_dir.iterdir()):
                if run_dir.is_dir() and (run_dir / "img").exists():
                    run_dirs.append(run_dir)

    print(f"实验根目录 : {exp_dir}")
    print(f"最大迭代   : {MAX_ITER}")
    print(f"强制重建   : {FORCE}")
    print(f"实验 run 数: {len(run_dirs)}")
    print("=" * 60)

    for i, run_dir in enumerate(run_dirs, 1):
        rel = run_dir.relative_to(exp_dir)
        print(f"\n[{i}/{len(run_dirs)}] {rel}")
        process_experiment_run(run_dir)

    print("\n" + "=" * 60)
    print("完成。")


if __name__ == "__main__":
    main()
