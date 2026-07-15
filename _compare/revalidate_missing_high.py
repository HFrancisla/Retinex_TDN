#!/usr/bin/env python3
"""
重新验证缺少 R_high 的实验：加载旧权重，用当前 evaluate() 重新保存验证图，
然后重新生成 decomposition_analysis.txt。

用法：
    .venv/bin/python _compare/revalidate_missing_high.py
"""

import os
import sys
import torch
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


def revalidate_one(exp_rel_path: str) -> bool:
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

    # 2. 确保 loss_cfg 包含 smooth_version（旧配置可能缺失）
    if "smooth_version" not in loss_cfg and loss_cfg.get("mode", "").endswith("_pixel"):
        loss_cfg = dict(loss_cfg)
        loss_cfg["smooth_version"] = "v1"
        print(f"  [FIX] Added smooth_version=v1 to loss_cfg (missing in old config)")

    # 3. 构建模型
    model = build_model(model_cfg, DEVICE)
    print(f"  Model: {model_cfg.get('name')} on {DEVICE}")

    # 4. 加载权重
    weights_path = exp_dir / "weights" / "best_model.pth"
    if not weights_path.is_file():
        weights_path = exp_dir / "weights" / "last_model.pth"
    if not weights_path.is_file():
        print(f"  [ERROR] No checkpoint found in weights/")
        return False
    load_weights(model, str(weights_path), DEVICE)

    # 5. 构建 val_loader（全分辨率）
    val_loader = build_val_loader(data_cfg)

    # 6. 构建损失函数
    loss_function = _build_loss_function(loss_cfg).to(DEVICE)

    # 7. 运行 evaluate，保存图像到 img/10000/
    #    使用相对路径保持与原训练一致的结构
    file_img_path = str(exp_dir / "img")
    os.makedirs(file_img_path, exist_ok=True)

    print(f"  Running evaluate() → save to img/{VALIDATION_ITER}/ ...")
    metrics, psnr = evaluate(
        model=model,
        data_loader=val_loader,
        device=DEVICE,
        lr=0.0,
        filefold_path=file_img_path,
        loss_function=loss_function,
        save_images=True,
        global_iter=VALIDATION_ITER,
        max_save_images=MAX_SAVE_IMAGES,
    )

    if metrics:
        print(f"  Val metrics: total_loss={metrics.get('total_loss', 'N/A'):.4f}")
    if psnr is not None:
        print(f"  Val PSNR:    {psnr:.2f} dB")

    # 8. 检查 R_high 是否成功写入
    img_iter_dir = exp_dir / "img" / str(VALIDATION_ITER)
    r_high_count = len(list(img_iter_dir.glob("*_R_high.png"))) if img_iter_dir.is_dir() else 0
    print(f"  R_high files saved: {r_high_count}")

    return r_high_count > 0


def main():
    print(f"Device: {DEVICE}")
    print(f"Re-validating {len(MISSING_EXPERIMENTS)} experiments...")

    success = []
    failed = []

    for exp_path in MISSING_EXPERIMENTS:
        try:
            if revalidate_one(exp_path):
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
                 str(exp_dir), "--iteration", str(VALIDATION_ITER), "--force"],
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
