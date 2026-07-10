#!/usr/bin/env python3
"""Retinex 合成 vs 原始输入 — 逐 iter 计算 avg PSNR。

S_low = R_low × L_low，应与原始 test/low 图像一致。PSNR 越高 = 分解越无损。
"""

import os, sys, yaml
from pathlib import Path
import cv2, numpy as np

ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
EXP_DIR = Path(os.environ.get("RETINEX_SYNTH_EXP_DIR", ROOT / "experiments"))


def process_run(run_dir: Path):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    data_root = ROOT / cfg["data"]["path"]
    test_files = sorted((data_root / "test" / "low").glob("*.*"))
    syn_dir = run_dir / "synthesis"
    if not syn_dir.exists():
        return

    out = run_dir / "synthesis_compare.txt"

    # ── 跳过机制：输出已存在且比所有 synthesis 子目录更新 → 跳过 ──
    if out.exists():
        out_mtime = out.stat().st_mtime
        newest_syn_mtime = 0
        for iter_dir in syn_dir.iterdir():
            if iter_dir.is_dir() and iter_dir.name.isdigit():
                newest_syn_mtime = max(newest_syn_mtime, iter_dir.stat().st_mtime)
        if newest_syn_mtime <= out_mtime:
            # 读取已有文件的第一条 PSNR 值用于摘要输出
            first_psnr = "-"
            for line in out.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    first_psnr = f"{float(parts[2]):.1f}"
                    break
            print(f"  [SKIP] {out.name} 已是最新  (PSNR={first_psnr} dB)")
            return

    with open(out, "w") as f:
        f.write(f"# {run_dir.name}\n# PSNR(dB) S_low vs test/low ({len(test_files)} images)\n\n")
        f.write(f"{'iter':>10}  {'n':>6}  {'PSNR_mean':>10}  {'PSNR_std':>8}\n")
        f.write(f"{'─'*10}  {'─'*6}  {'─'*10}  {'─'*8}\n")

        for iter_dir in sorted(syn_dir.iterdir(), key=lambda d: int(d.name)):
            if not iter_dir.is_dir():
                continue
            s_files = sorted(iter_dir.glob("*_S_low.png"))
            if not s_files:
                continue

            psnr = []
            for sf in s_files:
                idx = int(sf.stem.replace("_S_low", ""))
                if idx >= len(test_files):
                    break
                S = cv2.imread(str(sf))
                O = cv2.imread(str(test_files[idx]))
                if S.shape != O.shape:
                    S = cv2.resize(S, (O.shape[1], O.shape[0]))
                mse = np.mean((S.astype(float) - O.astype(float)) ** 2)
                psnr.append(20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0)

            line = f"{iter_dir.name:>10}  {len(psnr):>6}  {np.mean(psnr):>10.2f}  {np.std(psnr):>8.2f}\n"
            f.write(line)
        print(f"  -> {out.name}   PSNR={np.mean(psnr):.1f} dB")


def main():
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
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run.relative_to(EXP_DIR)}")
        process_run(run)
    print("done")


if __name__ == "__main__":
    main()
