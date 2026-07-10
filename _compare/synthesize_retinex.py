#!/usr/bin/env python3
"""Retinex 合成: S = R × L（逐像素相乘）。遍历 experiments/ 下 val 图像，验证分解-重建一致性。

R: 3 通道彩色反射图 [0,1]。L: 单通道光照图 [0,1]（保存时复制为 3 通道）。
    RetinexPointRaw → L 是标量，全图统一亮度系数。
    RetinexPixel*   → L 是逐像素光照图，暗处小值压暗、亮处大值提亮。

用法:
    .venv/bin/python3 scripts/synthesize_retinex.py                         # 全量
    RETINEX_SYNTH_MAX_ITER=10000 .venv/bin/python3 scripts/synthesize_retinex.py  # 仅 iter 10000

环境变量:
    RETINEX_SYNTH_MAX_ITER  最大迭代轮次 (默认 100000)
    RETINEX_SYNTH_EXP_DIR   实验根目录 (默认 ../experiments)

输出: img/ 同级 → synthesis/{iter}/，已存在跳过。
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# ============================================================
# 环境变量配置（脚本开头可修改）
# ============================================================
MAX_ITER: int = int(os.environ.get("RETINEX_SYNTH_MAX_ITER", "100000"))
"""最大处理的训练轮次。只处理目录名 ≤ 此值的迭代。"""

EXPERIMENTS_DIR: str = os.environ.get(
    "RETINEX_SYNTH_EXP_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"),
)
"""实验根目录路径。"""
# ============================================================


def synthesize_and_save(r_path: Path, l_path: Path, out_path: Path) -> bool:
    """
    Retinex 合成: S = R × L，保存为 PNG。

    Args:
        r_path: R 分量 (3-ch BGR, uint8, [0,255])
        l_path: L 分量 (3-ch BGR, uint8, [0,255]，三通道值相同)
        out_path: 输出路径

    Returns:
        True 如果实际合成了新文件，False 如果文件已存在被跳过
    """
    if out_path.exists():
        return False

    R = cv2.imread(str(r_path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    L = cv2.imread(str(l_path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0

    S = np.clip(R * L, 0.0, 1.0)
    S_uint8 = (S * 255.0).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), S_uint8)
    return True


def process_iteration(iter_dir: Path, out_iter_dir: Path) -> dict:
    """
    处理单个迭代目录：遍历所有 R 文件，配对 L 文件，合成 S = R × L。

    Returns:
        {"low": int, "high": int}  — 本次新合成的文件数（不含已存在的）
    """
    stats = {"low": 0, "high": 0}

    # ---- _low 分量 ----
    r_files_low = sorted(iter_dir.glob("*_R_low.png"))
    for r_file in r_files_low:
        idx = r_file.stem.replace("_R_low", "")
        l_file = iter_dir / f"{idx}_L_low.png"
        s_file = out_iter_dir / f"{idx}_S_low.png"

        if not l_file.exists():
            print(f"    [WARN] 缺少 L 文件: {l_file.name}")
            continue

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

        if synthesize_and_save(r_file, l_file, s_file):
            stats["high"] += 1

    return stats


def process_experiment_run(run_dir: Path) -> None:
    """处理单个实验 run 目录下所有 ≤ MAX_ITER 的迭代。"""
    img_dir = run_dir / "img"
    synthesis_dir = run_dir / "synthesis"

    if not img_dir.exists():
        print("  [SKIP] img 目录不存在")
        return

    iter_dirs = sorted(
        [d for d in img_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    iter_dirs = [d for d in iter_dirs if int(d.name) <= MAX_ITER]

    if not iter_dirs:
        print(f"  [SKIP] 无迭代目录 ≤ {MAX_ITER}")
        return

    total_low, total_high = 0, 0

    for iter_dir in iter_dirs:
        out_iter_dir = synthesis_dir / iter_dir.name

        # 快速检查：是否已全部完成
        expected_low = len(list(iter_dir.glob("*_R_low.png")))
        expected_high = len(list(iter_dir.glob("*_R_high.png")))
        existing_low = len(list(out_iter_dir.glob("*_S_low.png"))) if out_iter_dir.exists() else 0
        existing_high = len(list(out_iter_dir.glob("*_S_high.png"))) if out_iter_dir.exists() else 0

        if existing_low >= expected_low and existing_high >= expected_high:
            print(f"  [iter {iter_dir.name}] 已完成，跳过 ({existing_low} low + {existing_high} high)")
            total_low += existing_low
            total_high += existing_high
            continue

        stats = process_iteration(iter_dir, out_iter_dir)
        n_low, n_high = stats["low"], stats["high"]
        new_low = n_low if n_low > 0 else existing_low
        new_high = n_high if n_high > 0 else existing_high

        parts = [f"low={new_low}"]
        if expected_high > 0:
            parts.append(f"high={new_high}")
        print(f"  [iter {iter_dir.name}] " + ", ".join(parts))

        total_low += new_low
        total_high += new_high

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
