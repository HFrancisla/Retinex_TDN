#!/usr/bin/env python3
"""
构建最小可用的离线 HTML 副本，方便拷贝到 U 盘等移动设备上查看。

功能:
  1. 从 _compare/html/ 下的 HTML 中自动解析所有实验 run 及数据集信息
  2. 确定每个数据集的最佳 iter（覆盖 run 最多，与 HTML 自动选择一致）
  3. 仅复制前 N 张图像的 R / L / 重建R×L 图片到 _tmp/
  4. 改写 HTML 中 IMG_PREFIX 使图片路径变为相对路径，可直接在浏览器打开

要求:
  需先运行 _compare/generate_compare_html.py 和 generate_model_compare_html.py
  生成全部 compare*.html，本脚本从中读取实验元信息。

用法:
  python scripts/build_minimal.py                 # 默认复制前 5 张图像
  python scripts/build_minimal.py -n 10           # 复制前 10 张图像
  python scripts/build_minimal.py -n 3 -o /mnt/usb/offline  # 指定输出目录

输出目录结构:
  _tmp/
  ├── compare*.html            ← 所有对比页 (IMG_PREFIX 已改写)
  ├── datasets/                ← 各数据集 test/low 原图 (前 N 张)
  └── experiments/             ← 各 run 的 img/ 和 synthesis/ (前 N 张, 最佳 iter)

复用说明:
  新增实验后，重新生成 compare*.html，再运行本脚本即可自动包含新 run。
  脚本从 HTML 内嵌的 DATA 中动态解析，无需手动维护实验列表。
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML_DIR = ROOT / "_compare" / "html"
DEFAULT_OUT = ROOT / "_tmp"
DEFAULT_FIRST_N = 5


# ── HTML 数据解析 ─────────────────────────────────────────────

def parse_html_data(html_path: Path) -> dict:
    """从 HTML 中提取 JS const DATA = {...}; 返回解析后的 Python dict。"""
    text = html_path.read_text()
    # 优先匹配带后续变量声明的版本 (compare.html / model 级 HTML)
    m = re.search(r"const DATA = (\{.*?\})\s*;\s*\n\s*let currentDS", text, re.DOTALL)
    if not m:
        m = re.search(r"const DATA = (\{.*?\});", text, re.DOTALL)
    if not m:
        raise ValueError(f"无法解析 DATA: {html_path}")
    return json.loads(m.group(1))


def pick_best_iter(info: dict) -> str:
    """
    选择覆盖 run 最多的 iter。

    复刻 HTML JS 中的 auto-select 逻辑：
    遍历 all_iters，对每个 iter 统计有 img_indices 的 run 数，取最大值。
    """
    best_iter = info["all_iters"][0] if info["all_iters"] else "10000"
    max_count = 0

    # 展平嵌套结构 → 拿到所有 run 的列表
    # compare.html:       models → modes → losses → runs
    # 模型级 compare*.html: modes → losses → runs
    all_runs = []
    for top in info.get("models", info.get("modes", [])):
        for mid in top.get("modes", top.get("losses", [])):
            for loss in mid.get("losses", [mid]):
                if isinstance(loss, dict) and "runs" in loss:
                    all_runs.extend(loss["runs"])

    for it in info["all_iters"]:
        cnt = sum(1 for r in all_runs if it in r.get("img_indices", {}))
        if cnt > max_count:
            max_count = cnt
            best_iter = it
    return best_iter


def collect_all_runs():
    """
    从 compare.html（覆盖全部模型）提取所有实验 run 及数据集信息。

    Returns:
        dict like {dataset: {"iter": str, "test_files": [...], "runs": [(rel_path, iter), ...]}}
        失败返回 None
    """
    compare_html = HTML_DIR / "compare.html"
    if not compare_html.exists():
        print("[ERROR] compare.html 不存在，请先运行 _compare/generate_compare_html.py")
        return None

    data = parse_html_data(compare_html)
    result = {}

    for ds_key, info in data.items():
        best_iter = pick_best_iter(info)
        test_files = info.get("test_files", [])[:FIRST_N]
        runs = []
        for model in info.get("models", []):
            for mode in model.get("modes", []):
                for loss in mode.get("losses", []):
                    for r in loss["runs"]:
                        if best_iter in r.get("img_indices", {}):
                            runs.append((r["rel_path"], best_iter))

        result[ds_key] = {
            "iter": best_iter,
            "test_files": test_files,
            "runs": runs,
        }
        print(f"  {ds_key}: iter={best_iter}, test={len(test_files)}, runs={len(runs)}")

    return result


# ── 构建逻辑 ──────────────────────────────────────────────────

def build(out_dir: Path, first_n: int):
    global FIRST_N
    FIRST_N = first_n

    html_files = sorted(HTML_DIR.glob("compare*.html"))
    if not html_files:
        print("[ERROR] 未找到 HTML 文件，请先生成 _compare/html/compare*.html")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 从 compare.html 收集所有需要的文件 ──
    print("解析 compare.html 收集运行信息...")
    all_datasets = collect_all_runs()
    if not all_datasets:
        sys.exit(1)

    # ── 复制图像 ──
    total_copied = 0
    total_skipped = 0

    for ds, info in sorted(all_datasets.items()):
        best_iter = info["iter"]
        test_files = info["test_files"]
        runs = info["runs"]

        print(f"\n复制数据集: {ds}  (iter={best_iter})")

        # Test/low 原始图像
        for fname in test_files:
            src = ROOT / ds / "test" / "low" / fname
            dst = out_dir / ds / "test" / "low" / fname
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                total_copied += 1
            else:
                print(f"  [MISS] {src}")
                total_skipped += 1

        # 每个 run 的 R / L / S 图像（前 FIRST_N 张）
        for rel_path, it in runs:
            for idx in range(FIRST_N):
                # R & L from img/
                for suffix in ["_R_low.png", "_L_low.png"]:
                    src = ROOT / rel_path / "img" / it / f"{idx}{suffix}"
                    dst = out_dir / rel_path / "img" / it / f"{idx}{suffix}"
                    if src.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                        total_copied += 1
                    else:
                        total_skipped += 1

                # S (R×L 重建) from synthesis/
                src_s = ROOT / rel_path / "synthesis" / it / f"{idx}_S_low.png"
                dst_s = out_dir / rel_path / "synthesis" / it / f"{idx}_S_low.png"
                if src_s.exists():
                    dst_s.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_s, dst_s)
                    total_copied += 1
                else:
                    total_skipped += 1

    # ── 改写并复制 HTML ──
    print("\n--- 复制 HTML ---")
    for hf in html_files:
        text = hf.read_text()
        # 将 ../../ 前缀改为空 (HTML 在 out_dir 根目录)
        text = text.replace('const IMG_PREFIX = "../../";', 'const IMG_PREFIX = "";')
        # 添加版本标记
        text = text.replace("<title>", "<title>[精简版] ")
        text = text.replace(
            "| 点击图片放大",
            f"| 点击图片放大 | 精简离线版(仅{first_n}张)",
        )

        out_html = out_dir / hf.name
        out_html.write_text(text)
        print(f"  {hf.name} → {out_html}")

    # ── 统计 ──
    print(f"\n{'='*50}")
    print(f"完成: 复制 {total_copied} 个图像, 跳过 {total_skipped} 个 (源文件不存在)")
    print(f"输出: {out_dir}  ({len(html_files)} 个 HTML)")
    print(f"大小: {_format_size(out_dir)}")
    print(f"可直接用浏览器打开 {out_dir}/*.html")


def _format_size(path: Path) -> str:
    """格式化目录大小为可读字符串。"""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB"]:
        if total < 1024:
            return f"{total:.0f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="构建最小可用的离线 Retinex 对比 HTML 副本，方便拷贝到 U 盘查看",
    )
    parser.add_argument(
        "-n", "--num-images", type=int, default=DEFAULT_FIRST_N,
        help=f"每个 run 复制的图像张数 (默认: {DEFAULT_FIRST_N})",
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path, default=DEFAULT_OUT,
        help=f"输出目录 (默认: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    if args.num_images < 1:
        print("[ERROR] -n 必须 ≥ 1")
        sys.exit(1)

    build(out_dir=args.out_dir, first_n=args.num_images)


if __name__ == "__main__":
    main()
