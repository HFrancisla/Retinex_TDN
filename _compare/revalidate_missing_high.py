#!/usr/bin/env python3
"""
重新验证缺少 R_high 的实验：加载旧权重，用当前 evaluate() 重新保存验证图，
然后重新生成 decomposition_analysis.txt。

用法：
    .venv/bin/python _compare/revalidate_missing_high.py
    .venv/bin/python _compare/revalidate_missing_high.py \
        --experiment RetinexPixelClassic/paired/<run> --iteration 10000 --force
"""

import argparse
import datetime
import os
import re
import shutil
import sys
import torch
import yaml
from pathlib import Path

# ---- 添加项目根目录到 sys.path ----
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import MyDataSet, transforms as T
from models import RetinexPointRaw, RetinexPixelClassic, RetinexPixelTrans, RetinexPixelTransMinus
from utils import read_data, evaluate, load_config, _build_loss_function

# ---- 缺少 R_high 的实验（相对于 experiments/ 的路径）----
MISSING_EXPERIMENTS = [
    "RetinexPixelClassic/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.1smv2_20260712-150633",
    "RetinexPixelTrans/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.1smv2_20260712-163016",
    "RetinexPixelTrans/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.3smv2_20260712-213703",
    "RetinexPixelTransMinus/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.1smv2_20260712-180829",
    "RetinexPixelTrans/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.1smv1_20260710-212842",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALIDATION_ITER = 10000  # 统一使用 iter=10000 保存
MAX_SAVE_IMAGES = 250


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, default=VALIDATION_ITER)
    parser.add_argument(
        "--force", action="store_true",
        help="原子替换已有 img/<iteration>；旧目录会保留为带时间戳的备份",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help=(
            "仅重验证指定的实验（相对于 experiments/）；可重复传入。"
            "未指定时使用脚本内的历史缺失列表"
        ),
    )
    return parser.parse_args()


def build_model(model_cfg: dict, device: torch.device):
    """根据配置构建模型。"""
    name = model_cfg.get("name", "RetinexPointRaw")
    if name == "RetinexPixelClassic":
        model = RetinexPixelClassic(
            dim=model_cfg.get("dim", 24),
            l_channel=model_cfg.get("l_channel", 32),
        )
    elif name == "RetinexPixelTrans":
        model = RetinexPixelTrans(
            dim=model_cfg.get("dim", 24),
            l_heads=model_cfg.get("l_heads", 1),
            l_ffn_expansion=model_cfg.get("l_ffn_expansion", 2.66),
        )
    elif name == "RetinexPixelTransMinus":
        model = RetinexPixelTransMinus(
            dim=model_cfg.get("dim", 24),
            l_heads=model_cfg.get("l_heads", 1),
            l_ffn_expansion=model_cfg.get("l_ffn_expansion", 2.66),
        )
    elif name == "RetinexPointRaw":
        model = RetinexPointRaw(dim=model_cfg.get("dim", 24))
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device)


def load_weights(model, weights_path: str, device: torch.device):
    """加载 best_model.pth 权重。"""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    weights_dict = checkpoint.get("model", checkpoint)
    result = model.load_state_dict(weights_dict, strict=True)
    print(f"  Loaded weights: {result}")
    return checkpoint


def select_checkpoint(exp_dir: Path, requested_step: int, device: torch.device):
    """Select a checkpoint whose embedded global_iter exactly matches the label."""
    weight_dir = exp_dir / "weights"
    preferred = [weight_dir / "last_model.pth"]
    preferred.extend(sorted(weight_dir.glob(f"checkpoint_{requested_step}_*.pth")))
    preferred.extend([weight_dir / "best_model.pth", weight_dir / "final_best_model.pth"])
    seen = set()
    for path in preferred:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if int(checkpoint.get("global_iter", -1)) == requested_step:
            return path, checkpoint
    available = []
    for path in sorted(weight_dir.glob("*.pth")):
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            available.append(f"{path.name}:step={checkpoint.get('global_iter')}")
        except Exception:
            available.append(f"{path.name}:unreadable")
    raise FileNotFoundError(
        f"no checkpoint with global_iter={requested_step}; available={available}"
    )


def recover_loss_config(loss_cfg: dict, checkpoint: dict, run_name: str) -> dict:
    """Recover historic loss metadata without silently changing smooth semantics."""
    result = dict(loss_cfg)
    checkpoint_loss = checkpoint.get("config", {}).get("loss", {})
    for key, value in checkpoint_loss.items():
        result.setdefault(key, value)
    if "smooth_version" not in result and result.get("mode", "").endswith("_pixel"):
        match = re.search(r"sm(v[123])(?:_|$)", run_name)
        if not match:
            raise ValueError(
                "historic pixel config lacks smooth_version and run name does not encode it"
            )
        result["smooth_version"] = match.group(1)
        print(f"  [RECOVER] smooth_version={match.group(1)} from run name")
    return result


def build_val_loader(data_cfg: dict):
    """构建全分辨率验证 DataLoader（val_full transform = ToTensor only）。"""
    data_path = data_cfg.get("path", "")
    data_mode = data_cfg["mode"]
    val_batch_size = data_cfg.get("val_batch_size", 1)

    _, _, val_low_path, val_high_path = read_data(data_path, mode=data_mode)

    val_transform = T.Compose([T.ToTensor()])
    val_dataset = MyDataSet(
        images_low_path=val_low_path,
        images_high_path=val_high_path,
        transform=val_transform,
    )

    print(f"  Val samples: {len(val_dataset)} (low={len(val_low_path)}, high={len(val_high_path)})")

    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=DEVICE.type == "cuda",
        num_workers=0,
        collate_fn=val_dataset.collate_fn,
    )


def revalidate_one(exp_rel_path: str, requested_step: int, force: bool) -> bool:
    """对一个实验重新验证并返回是否成功。"""
    exp_dir = ROOT / "experiments" / exp_rel_path
    print(f"\n{'='*70}")
    print(f"Re-validating: {exp_rel_path}")
    print(f"Directory:     {exp_dir}")
    print(f"{'='*70}")

    # 1. 加载配置
    config_path = exp_dir / "config.yaml"
    if not config_path.is_file():
        print(f"  [ERROR] config.yaml not found: {config_path}")
        return False
    cfg = load_config(str(config_path))
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})

    # Select before building the loss so historic metadata can be recovered from
    # the checkpoint itself.  Never label a step-9500 model as img/10000.
    weights_path, checkpoint = select_checkpoint(exp_dir, requested_step, DEVICE)
    loss_cfg = recover_loss_config(loss_cfg, checkpoint, exp_dir.name)

    # 3. 构建模型
    model = build_model(model_cfg, DEVICE)
    print(f"  Model: {model_cfg.get('name')} on {DEVICE}")

    # 4. 加载权重
    loaded = load_weights(model, str(weights_path), DEVICE)
    checkpoint_step = int(loaded["global_iter"])
    if checkpoint_step != requested_step:
        raise ValueError(f"checkpoint step {checkpoint_step} != requested {requested_step}")

    # 5. 构建 val_loader（全分辨率）
    val_loader = build_val_loader(data_cfg)

    # 6. 构建损失函数
    loss_function = _build_loss_function(loss_cfg).to(DEVICE)

    # 7. Render into a temporary root, then publish atomically.  Existing
    # results are never merged with newly rendered files.
    target_dir = exp_dir / "img" / str(checkpoint_step)
    if target_dir.exists() and not force:
        print(f"  [SKIP] target exists; pass --force to replace safely: {target_dir}")
        return len(list(target_dir.glob("*_R_high.png"))) > 0
    temp_root = exp_dir / "img" / ".revalidate_tmp"
    shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    print(f"  Running evaluate() → temporary render for step {checkpoint_step} ...")
    metrics, psnr = evaluate(
        model=model,
        data_loader=val_loader,
        device=DEVICE,
        lr=0.0,
        filefold_path=str(temp_root),
        loss_function=loss_function,
        save_images=True,
        global_iter=checkpoint_step,
        max_save_images=MAX_SAVE_IMAGES,
    )

    if metrics:
        print(f"  Val metrics: total_loss={metrics.get('total_loss', 'N/A'):.4f}")
    if psnr is not None:
        print(f"  Val PSNR:    {psnr:.2f} dB")

    rendered_dir = temp_root / str(checkpoint_step)
    r_high_count = len(list(rendered_dir.glob("*_R_high.png"))) if rendered_dir.is_dir() else 0
    print(f"  R_high files saved: {r_high_count}")
    if r_high_count == 0:
        shutil.rmtree(temp_root, ignore_errors=True)
        return False

    backup_dir = None
    if target_dir.exists():
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_dir = target_dir.with_name(f"{target_dir.name}.backup-{timestamp}")
        os.replace(target_dir, backup_dir)
    os.replace(rendered_dir, target_dir)
    shutil.rmtree(temp_root, ignore_errors=True)

    # Any pre-existing synthesis now refers to the old components and must not
    # survive under the same iteration label.
    stale_synthesis = exp_dir / "synthesis" / str(checkpoint_step)
    if stale_synthesis.exists():
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        stale_backup = stale_synthesis.with_name(
            f"{stale_synthesis.name}.stale-{timestamp}"
        )
        os.replace(stale_synthesis, stale_backup)
        print(f"  Moved stale synthesis to: {stale_backup}")

    provenance = {
        "rendered_at": datetime.datetime.now().isoformat(),
        "checkpoint": str(weights_path.relative_to(exp_dir)),
        "checkpoint_step": checkpoint_step,
        "published_image_dir": str(target_dir.relative_to(exp_dir)),
        "backup_image_dir": str(backup_dir.relative_to(exp_dir)) if backup_dir else None,
        "smooth_version": loss_cfg.get("smooth_version"),
        "metrics": metrics,
        "psnr_proxy": psnr,
    }
    (exp_dir / f"revalidation_step_{checkpoint_step}.yaml").write_text(
        yaml.safe_dump(provenance, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return True


def main():
    args = parse_args()
    experiments = args.experiments or MISSING_EXPERIMENTS
    print(f"Device: {DEVICE}")
    print(f"Re-validating {len(experiments)} experiments...")

    success = []
    failed = []

    for exp_path in experiments:
        try:
            if revalidate_one(exp_path, args.iteration, args.force):
                success.append(exp_path)
            else:
                failed.append(exp_path)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            failed.append(exp_path)

    # ---- 汇总 ----
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(success)} succeeded, {len(failed)} failed")
    print(f"{'='*70}")

    if success:
        print("\nNow re-running decomposition analysis on succeeded experiments...")
        venv_python = ROOT / ".venv" / "bin" / "python"
        analyze_script = ROOT / "_compare" / "analyze_decomposition.py"
        import subprocess
        for exp_path in success:
            exp_dir = ROOT / "experiments" / exp_path
            print(f"\n  Analyzing: {exp_dir.name}")
            result = subprocess.run(
                [str(venv_python), str(analyze_script),
                 str(exp_dir), "--iteration", str(args.iteration), "--force"],
                capture_output=True, text=True, cwd=str(ROOT),
            )
            print(f"  {result.stdout.strip()}")
            if result.stderr.strip():
                print(f"  [stderr] {result.stderr.strip()}")

    if failed:
        print(f"\nFailed experiments:")
        for f in failed:
            print(f"  - {f}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
