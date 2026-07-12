#!/usr/bin/env python3
"""
scripts/smoke_test.py — 冒烟测试：每个类别跑 5 个 training step，验证模型/损失/数据能正确对接。

每次测试用独立子进程运行 train.py，并自动从原数据集建 symlink 轻量子集（只取前 20 张图），
避免全量 DataLoader 初始化开销和超时导致的 GPU 残留。

用法:
    .venv/bin/python scripts/smoke_test.py                  # 全部子集
    .venv/bin/python scripts/smoke_test.py --subset lolv2_paired
    .venv/bin/python scripts/smoke_test.py --list           # 列出可用子集
退出码 0 = 全部通过，非 0 = 有失败。
"""

import os, sys, yaml, subprocess, tempfile, shutil, argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
PYTHON = os.path.join(ROOT_DIR, ".venv", "bin", "python")
TRAIN_SCRIPT = os.path.join(ROOT_DIR, "train.py")

SMOKE_STEPS = 5
SMOKE_MAX_IMAGES = 20   # 每 split 只取前 N 张图
SMOKE_TIMEOUT = 120

# ── 按批次定义的冒烟子集 ──────────────────────────────────────────────
SMOKE_SUBSETS = {
    "lolv2_paired": [
        ("paired_point",
         "configs/RetinexPointRaw/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq.yaml"),
        ("paired_pixel",
         "configs/RetinexPixelClassic/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.1sm.yaml"),
    ],
    "lolv2_pure_single": [
        ("pure_low_single_point",
         "configs/RetinexPointRaw/pure_low_single/LOLv2_1.0r_0.05anchorv2_0.05bdsp.yaml"),
        ("pure_low_single_pixel",
         "configs/RetinexPixelTrans/pure_low_single/LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1sm.yaml"),
        ("pure_low_single_pixel_minus",
         "configs/RetinexPixelTransMinus/pure_low_single/LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.0sm.yaml"),
    ],
    "lolv2_ablation": [
        ("ablation_lolv2",
         "configs/RetinexPixelTrans/pure_low_single/LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1sm.yaml"),
    ],
    "bdd": [
        ("BDD_pure_single_point",
         "configs/RetinexPointRaw/pure_low_single/BDD_1.0r_0.05anchorv2_0.05bdsp.yaml"),
        ("BDD_pure_single_pixel",
         "configs/RetinexPixelTrans/pure_low_single/BDD_1.0r_0.05anchorv2_0.05bdsp_0.1sm.yaml"),
    ],
}


# ── 数据子集构建 ────────────────────────────────────────────────────

def _build_smoke_subset(src_root, max_images, tmpdir):
    """在 tmpdir 下创建源数据集的 symlink 轻量子集（每 split 只取前 N 张）。

    自动适配各种目录约定：
      - train/low + train/high  （LOLv2 配对）
      - train/images            （BDD 纯 low，fallback 到 images 目录）
      - val → test fallback 由 read_data / read_pure_low_data 在运行时处理
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    src_root = os.path.abspath(src_root)
    subset_root = os.path.join(tmpdir, "smoke_data")
    os.makedirs(subset_root, exist_ok=True)

    def _image_names(d):
        if not os.path.isdir(d):
            return []
        return sorted(
            f for f in os.listdir(d)
            if os.path.splitext(f)[1].lower() in img_exts
        )

    def _symlink(src_dir, dst_dir, names):
        os.makedirs(os.path.abspath(dst_dir), exist_ok=True)
        for name in names[:max_images]:
            s = os.path.join(os.path.abspath(src_dir), name)
            d = os.path.join(os.path.abspath(dst_dir), name)
            if not os.path.exists(d):
                os.symlink(s, d)

    for split in ["train", "val", "test"]:
        s = os.path.join(src_root, split)
        if not os.path.isdir(s):
            continue
        t = os.path.join(subset_root, split)
        for sub in ["low", "high", "images"]:
            src_sub = os.path.join(s, sub)
            if os.path.isdir(src_sub):
                _symlink(src_sub, os.path.join(t, sub), _image_names(src_sub))

    return subset_root


# ── 单测试执行 ──────────────────────────────────────────────────────

def smoke_one(label, config_path):
    """覆盖 config → 建数据子集 → Popen 跑 train.py → 清理。"""
    from utils import load_config
    config_abs = os.path.join(ROOT_DIR, config_path)
    cfg = load_config(config_abs)

    # ── 训练参数覆写 ──
    cfg["training"]["max_iterations"] = SMOKE_STEPS
    cfg["training"]["log_interval"] = SMOKE_STEPS + 1
    cfg["training"]["warmup_iterations"] = 0
    cfg["training"]["save_ckpt_interval"] = 0
    cfg["training"]["eval"]["eval_interval"] = SMOKE_STEPS + 1
    cfg["training"]["eval"]["save_img_interval"] = 0
    cfg["experiment"]["auto_name"] = False
    cfg["experiment"]["name"] = "_smoke_test_"

    tmpdir = tempfile.mkdtemp(prefix="smoke_")
    tmp_config = os.path.join(tmpdir, "config.yaml")

    # ── 数据子集：只取前 20 张图，避免全量加载 ──
    src_path = cfg["data"].get("path", "")
    if src_path and os.path.isdir(src_path):
        subset_root = _build_smoke_subset(src_path, SMOKE_MAX_IMAGES, tmpdir)
        cfg["data"]["path"] = subset_root
    cfg["data"]["num_workers"] = 0

    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f)

    # ── 执行 ──
    proc = None
    try:
        proc = subprocess.Popen(
            [PYTHON, TRAIN_SCRIPT, "--config", tmp_config],
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        stdout, stderr = proc.communicate(timeout=SMOKE_TIMEOUT)
        if proc.returncode != 0:
            for line in stderr.strip().split("\n")[-15:]:
                print(f"      {line}")
            return False
        return True

    except subprocess.TimeoutExpired:
        print(f"      TIMEOUT (>{SMOKE_TIMEOUT}s)")
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        return False

    except KeyboardInterrupt:
        print("\n      INTERRUPTED — cleaning up subprocess...")
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise

    finally:
        # 清理临时数据和实验产物
        smoke_exp = os.path.join(ROOT_DIR, "experiments")
        if os.path.isdir(smoke_exp):
            for sub in os.listdir(smoke_exp):
                subp = os.path.join(smoke_exp, sub)
                if os.path.isdir(subp):
                    for sub2 in os.listdir(subp):
                        if "_smoke_test_" in sub2:
                            shutil.rmtree(os.path.join(subp, sub2), ignore_errors=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── CLI ─────────────────────────────────────────────────────────────

def list_subsets():
    print("Available smoke test subsets:")
    for name, tests in SMOKE_SUBSETS.items():
        print(f"  {name:<24s}  ({len(tests)} tests)")
    all_unique = len({cfg for tests in SMOKE_SUBSETS.values() for _, cfg in tests})
    print(f"\n  {'all':<24s}  ({all_unique} unique tests — default)")


def main():
    parser = argparse.ArgumentParser(description="Smoke test: 5-step training for each config category")
    parser.add_argument("--subset", type=str, default=None,
                        help="Run only a specific subset (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List available subsets and exit")
    args = parser.parse_args()

    if args.list:
        list_subsets()
        return 0

    if args.subset is None:
        seen = set()
        test_cases = []
        for name, tests in SMOKE_SUBSETS.items():
            for label, cfg in tests:
                if cfg not in seen:
                    seen.add(cfg)
                    test_cases.append((f"{name}/{label}", cfg))
    elif args.subset not in SMOKE_SUBSETS:
        print(f"Unknown subset: {args.subset}", file=sys.stderr)
        list_subsets()
        return 2
    else:
        test_cases = [(label, cfg) for label, cfg in SMOKE_SUBSETS[args.subset]]

    total = len(test_cases)
    failed = 0
    print(f"\n  Smoke test: {total} cases × {SMOKE_STEPS} steps each "
          f"(subset: {args.subset or 'all'})\n")

    try:
        for label, config_path in test_cases:
            print(f"  [{label}] ", end="", flush=True)
            ok = smoke_one(label, config_path)
            if ok:
                print("✅")
            else:
                print("❌ FAIL")
                failed += 1
    except KeyboardInterrupt:
        print("\n\n  ⚠  Interrupted by user. If WSL becomes unresponsive:")
        print("       wsl --shutdown   (from Windows PowerShell)")
        return 130

    print(f"\n  Result: {total - failed}/{total} passed, {failed} failed\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
