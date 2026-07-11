"""
utils.py

训练与评估辅助函数。

包含数据路径读取、单 epoch 训练/验证、学习率 warmup 调度器、
结果可视化保存等工具函数。
"""

import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import numpy as np
import cv2

from loss import PairedLoss, UnpairedLoss, PureLowSingleLoss, PureLowDoubleLoss

def calculate_psnr(pred, target, max_val=1.0):
    """计算 PSNR (Peak Signal-to-Noise Ratio)。
    
    Args:
        pred: 预测图像张量 [B, C, H, W]，范围 [0, max_val]
        target: 目标图像张量 [B, C, H, W]，范围 [0, max_val]
        max_val: 像素最大值（默认 1.0）
    
    Returns:
        PSNR 值 (dB)
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10.0 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()




def load_config(config_path: str) -> dict:
    """
    加载 YAML 配置文件。

    Args:
        config_path: YAML 配置文件路径

    Returns:
        dict: 配置字典
    """
    import yaml

    assert os.path.exists(config_path), f"Config file: '{config_path}' does not exist."

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)


    return cfg


# 各模式对应的有效 loss 字段（用于配置校验）
_VALID_LOSS_FIELDS = {
    'paired_point': {
        'recon_weight_high', 'recon_weight_low',
        'cross_recon_weight_low', 'cross_recon_weight_high',
        'equal_r_weight',
    },
    'paired_pixel': {
        'recon_weight_high', 'recon_weight_low',
        'cross_recon_weight_low', 'cross_recon_weight_high',
        'smooth_weight', 'equal_r_weight',
    },
    'unpaired_point': {
        'recon_weight', 'anchor_weight', 'bdsp_weight',
        'self_recon_weight',
    },
    'unpaired_pixel': {
        'recon_weight', 'anchor_weight', 'bdsp_weight',
        'smooth_weight', 'self_recon_weight',
    },
    'pure_low_double_point': {
        'recon_weight', 'anchor_weight', 'bdsp_weight',
        'self_recon_weight', 'reflect_weight',
    },
    'pure_low_double_pixel': {
        'recon_weight', 'anchor_weight', 'bdsp_weight',
        'smooth_weight', 'self_recon_weight', 'reflect_weight',
    },
    'pure_low_single_point': {
        'recon_weight', 'anchor_weight', 'bdsp_weight',
    },
    'pure_low_single_pixel': {
        'recon_weight', 'anchor_weight', 'bdsp_weight',
        'smooth_weight',
    },
}

_VALID_MODES = sorted(_VALID_LOSS_FIELDS.keys())


def _build_loss_function(loss_cfg):
    """根据 loss_cfg 构建损失模块。

    loss_cfg 中的 'mode' 字段必须为以下 8 种之一：
      paired_point, paired_pixel,
      unpaired_point, unpaired_pixel,
      pure_low_double_point, pure_low_double_pixel,
      pure_low_single_point, pure_low_single_pixel

    mode 格式: {data_mode}_{l_type}，其中 l_type 为 point 或 pixel。
    仅允许当前模式支持的字段，多余字段会报错。
    """
    cfg = dict(loss_cfg or {})
    mode = cfg.pop('mode', None)

    if mode is None:
        raise ValueError(
            "loss.mode must be explicitly set. Choose from: "
            + ", ".join(_VALID_MODES)
        )
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid loss.mode='{mode}'. Choose from: "
            + ", ".join(_VALID_MODES)
        )

    # 解析 mode -> data_mode + l_type
    if mode.endswith('_point'):
        data_mode = mode[:-len('_point')]
        l_type = 'point'
    else:
        data_mode = mode[:-len('_pixel')]
        l_type = 'pixel'

    cfg['l_type'] = l_type

    # 校验：所有必填字段必须显式声明
    required = _VALID_LOSS_FIELDS[mode]  # 不含 l_type，它由代码自动注入
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(
            f"loss.mode='{mode}' 要求显式声明以下字段，但 config 中缺失：\n"
            + "\n".join(f"  {f}" for f in sorted(missing))
            + f"\n\n请参考 _VALID_LOSS_FIELDS['{mode}'] = {sorted(required)} 补全。"
        )

    # 拒绝无效字段（l_type 始终保留）
    valid = required | {'l_type'}
    extra = set(cfg.keys()) - valid
    if extra:
        raise ValueError(
            f"loss.mode='{mode}' 不支持以下字段：\n"
            + "\n".join(f"  {f}" for f in sorted(extra))
            + f"\n\n当前模式仅支持：{sorted(required)}。"
        )

    if data_mode == 'paired':
        return PairedLoss(**cfg)
    if data_mode == 'pure_low_single':
        return PureLowSingleLoss(**cfg)
    if data_mode == 'pure_low_double':
        return PureLowDoubleLoss(**cfg)
    if data_mode == 'unpaired':
        return UnpairedLoss(**cfg)

    raise ValueError(f"Cannot parse data_mode from mode='{mode}'")


def read_data(root: str, mode: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "test")
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

    train_images_low_path = []
    train_images_high_path = []
    val_images_low_path = []
    val_images_high_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]
    train_high_root = os.path.join(train_root, "high")
    train_low_root = os.path.join(train_root, "low")

    val_high_root = os.path.join(val_root, "high")
    val_low_root = os.path.join(val_root, "low")

    train_low_path = sorted(
        [os.path.join(train_low_root, i) for i in os.listdir(train_low_root)
         if os.path.splitext(i)[-1] in supported]
    )
    train_high_path = sorted(
        [os.path.join(train_high_root, i) for i in os.listdir(train_high_root)
         if os.path.splitext(i)[-1] in supported]
    )

    val_low_path = sorted(
        [os.path.join(val_low_root, i) for i in os.listdir(val_low_root)
         if os.path.splitext(i)[-1] in supported]
    )
    val_high_path = sorted(
        [os.path.join(val_high_root, i) for i in os.listdir(val_high_root)
         if os.path.splitext(i)[-1] in supported]
    )

    if mode == "paired":
        assert len(train_low_path) == len(train_high_path), "The length of train dataset does not match. low:{}, high:{}".format(len(train_low_path), len(train_high_path))
        assert len(val_low_path) == len(val_high_path), "The length of val dataset does not match. low:{}, high:{}".format(len(val_low_path), len(val_high_path))
        print("image pair check finish")
    else:
        if mode in ("pure_low_single", "pure_low_double"):
            print("pure_low mode: low-only training, train({}), val({})".format(len(train_low_path), len(val_low_path)))
        else:
            print("unpaired mode: low({}) and high({}) loaded independently".format(len(train_low_path), len(train_high_path)))
    train_images_low_path = list(train_low_path)
    train_images_high_path = list(train_high_path)
    val_images_low_path = list(val_low_path)
    val_images_high_path = list(val_high_path)


    total_dataset_nums = len(train_low_path) + len(train_high_path) + len(val_low_path) + len(val_high_path)
    print("{} images were found in the dataset.".format(total_dataset_nums))
    print("{} low light images for training.".format(len(train_low_path)))
    print("{} normal light images for training ref.".format(len(train_high_path)))
    print("{} low light images for validation.".format(len(val_low_path)))
    print("{} normal light images for validation ref.".format(len(val_high_path)))

    return train_images_low_path, train_images_high_path, val_images_low_path, val_images_high_path





def read_pure_low_data(root: str):
    """读取 pure_low 数据集，仅使用 low 文件夹。"""
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "test")
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    # 兼容性：如果不存在 low/high 子目录，直接使用 images 目录
    train_low_root = os.path.join(train_root, "low")
    if not os.path.exists(train_low_root):
        images_root = os.path.join(train_root, "images")
        if os.path.exists(images_root):
            print(f"[read_pure_low_data] '{train_low_root}' not found, fallback to '{images_root}'")
            train_low_root = images_root
        else:
            raise FileNotFoundError(
                f"Neither '{train_low_root}' nor '{images_root}' exists."
            )

    val_low_root = os.path.join(val_root, "low")
    if not os.path.exists(val_low_root):
        images_root = os.path.join(val_root, "images")
        if os.path.exists(images_root):
            print(f"[read_pure_low_data] '{val_low_root}' not found, fallback to '{images_root}'")
            val_low_root = images_root
        else:
            raise FileNotFoundError(
                f"Neither '{val_low_root}' nor '{images_root}' exists."
            )

    train_low_path = sorted(
        [os.path.join(train_low_root, i) for i in os.listdir(train_low_root)
         if os.path.splitext(i)[-1] in supported]
    )
    val_low_path = sorted(
        [os.path.join(val_low_root, i) for i in os.listdir(val_low_root)
         if os.path.splitext(i)[-1] in supported]
    )

    print("pure_low loader: train({}), val({})".format(len(train_low_path), len(val_low_path)))
    return list(train_low_path), list(val_low_path)
def train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch, loss_cfg=None):
    model.train()
    loss_function = _build_loss_function(loss_cfg)

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_recon_loss = torch.zeros(1).to(device)
    accu_anchor_loss = torch.zeros(1).to(device)
    accu_bdsp_loss = torch.zeros(1).to(device)
    accu_smooth_loss = torch.zeros(1).to(device)
    accu_self_recon_loss = torch.zeros(1).to(device)
    accu_psnr = 0.0
    psnr_count = 0

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    lr = optimizer.param_groups[0]["lr"]
    for step, data in enumerate(data_loader):
        if isinstance(loss_function, PureLowSingleLoss):
            I_low = data
            if torch.cuda.is_available():
                I_low = I_low.to(device)

            R_low, L_low = model(I_low)

            loss, loss_recon, loss_anchor, loss_bdsp, loss_smooth, loss_self_recon = \
                loss_function(R_low, L_low, I_low)
        else:
            I_low, I_high = data

            if torch.cuda.is_available():
                I_low = I_low.to(device)
                I_high = I_high.to(device)

            R_low, L_low = model(I_low)
            R_high, L_high = model(I_high)

            # Skip extra forward pass when self_recon_weight == 0
            if loss_function.self_recon_weight > 0:
                _, L_R_low = model(R_low)
                _, L_R_high = model(R_high)
            else:
                L_R_low = None
                L_R_high = None

            loss, loss_recon, loss_anchor, loss_bdsp, loss_smooth, loss_self_recon = \
                loss_function(R_low, R_high, L_low, L_high, I_low, I_high, L_R_low, L_R_high)

        loss.backward()

        accu_total_loss += loss.detach()
        accu_recon_loss += loss_recon.detach()
        accu_anchor_loss += loss_anchor.detach()
        accu_bdsp_loss += loss_bdsp.detach()
        accu_smooth_loss += loss_smooth.detach()
        accu_self_recon_loss += loss_self_recon.detach()

        lr = optimizer.param_groups[0]["lr"]

        n = step + 1
        data_loader.desc = (
            "[train epoch {}] total: {:.3f}  recon: {:.3f}  anchor: {:.3f}  "
            "bdsp: {:.3f}  smooth: {:.3f}  self-recon: {:.3f}  lr: {:.6f}"
        ).format(
            epoch,
            accu_total_loss.item() / n, accu_recon_loss.item() / n,
            accu_anchor_loss.item() / n, accu_bdsp_loss.item() / n,
            accu_smooth_loss.item() / n, accu_self_recon_loss.item() / n, lr
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    step_count = max(step + 1, 1) if data_loader.iterable else 0
    if step_count == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lr)

    return (accu_total_loss.item() / step_count, accu_recon_loss.item() / step_count,
            accu_anchor_loss.item() / step_count, accu_bdsp_loss.item() / step_count,
            accu_smooth_loss.item() / step_count, accu_self_recon_loss.item() / step_count, lr)


def train_step(model, optimizer, loss_function, data, device, lr_scheduler):
    """单步训练：前向 + 反向 + 更新，返回各项损失。"""
    model.train()

    if isinstance(loss_function, PureLowSingleLoss):
        I_low = data
        if torch.cuda.is_available():
            I_low = I_low.to(device)

        R_low, L_low = model(I_low)
        loss, loss_recon, loss_anchor, loss_bdsp, loss_smooth, loss_self_recon = \
            loss_function(R_low, L_low, I_low)
    else:
        I_low, I_high = data
        if torch.cuda.is_available():
            I_low = I_low.to(device)
            I_high = I_high.to(device)

        R_low, L_low = model(I_low)
        R_high, L_high = model(I_high)

        if loss_function.self_recon_weight > 0:
            _, L_R_low = model(R_low)
            _, L_R_high = model(R_high)
        else:
            L_R_low = None
            L_R_high = None

        loss, loss_recon, loss_anchor, loss_bdsp, loss_smooth, loss_self_recon = \
            loss_function(R_low, R_high, L_low, L_high, I_low, I_high, L_R_low, L_R_high)

    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    if not torch.isfinite(loss):
        print('WARNING: non-finite loss, ending training ', loss)
        sys.exit(1)

    return (loss.item(), loss_recon.item(), loss_anchor.item(),
            loss_bdsp.item(), loss_smooth.item(), loss_self_recon.item())


@torch.no_grad()
def evaluate(model, data_loader, device, lr, filefold_path,
             loss_function=None, loss_cfg=None, save_images=False, global_iter=0,
             max_save_images=250):

    if loss_function is None:
        loss_function = _build_loss_function(loss_cfg)

    model.eval()

    accu_total_loss = torch.zeros(1).to(device)
    accu_recon_loss = torch.zeros(1).to(device)
    accu_anchor_loss = torch.zeros(1).to(device)
    accu_bdsp_loss = torch.zeros(1).to(device)
    accu_smooth_loss = torch.zeros(1).to(device)
    accu_self_recon_loss = torch.zeros(1).to(device)
    accu_psnr = 0.0
    psnr_count = 0
    sample_count = 0

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    if save_images:
        evalfold_path = os.path.join(filefold_path, str(global_iter))
        os.makedirs(evalfold_path, exist_ok=True)

    save_count = 0  # 已保存图像计数，达到 max_save_images 后停止保存

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        # --- 统一提取 low 图像并分解（所有模式一致）---
        if isinstance(data, (tuple, list)):
            I_low = data[0]
        else:
            I_low = data
        if torch.cuda.is_available():
            I_low = I_low.to(device)

        R_low, L_low = model(I_low)

        if save_images and save_count < max_save_images:
            R_low_img = tensor2numpy_R(R_low)
            L_low_img = tensor2numpy_L(L_low)
            save_pic(R_low_img, evalfold_path, str(step) + "_R_low")
            save_pic(L_low_img, evalfold_path, str(step) + "_L_low")
            save_count += 1

        # --- 损失计算（按模式分支）---
        if isinstance(loss_function, PureLowSingleLoss):
            loss, loss_recon, loss_anchor, loss_bdsp, loss_smooth, loss_self_recon = \
                loss_function(R_low, L_low, I_low)
        else:
            if isinstance(data, (tuple, list)):
                I_high = data[1]
            else:
                I_high = data
            if torch.cuda.is_available():
                I_high = I_high.to(device)

            R_high, L_high = model(I_high)

            # Skip extra forward pass when self_recon_weight == 0
            if loss_function.self_recon_weight > 0:
                _, L_R_low = model(R_low)
                _, L_R_high = model(R_high)
            else:
                L_R_low = None
                L_R_high = None

            loss, loss_recon, loss_anchor, loss_bdsp, loss_smooth, loss_self_recon = \
                loss_function(R_low, R_high, L_low, L_high, I_low, I_high, L_R_low, L_R_high)

            with torch.no_grad():
                psnr_val = calculate_psnr(R_low.clamp(0, 1), I_high.clamp(0, 1))
                batch_size = I_low.shape[0]
                accu_psnr += psnr_val * batch_size
                psnr_count += batch_size

        batch_size = I_low.shape[0]

        # loss 函数返回 batch 内均值，因此按 batch 样本数加权，确保最后一个
        # 不足 batch_size 的 batch 不会与完整 batch 获得相同权重。
        accu_total_loss += loss * batch_size
        accu_recon_loss += loss_recon * batch_size
        accu_anchor_loss += loss_anchor * batch_size
        accu_bdsp_loss += loss_bdsp * batch_size
        accu_smooth_loss += loss_smooth * batch_size
        accu_self_recon_loss += loss_self_recon * batch_size
        sample_count += batch_size

        avg_vals = [accu_total_loss.item() / sample_count,
                    accu_recon_loss.item() / sample_count,
                    accu_anchor_loss.item() / sample_count,
                    accu_bdsp_loss.item() / sample_count,
                    accu_smooth_loss.item() / sample_count,
                    accu_self_recon_loss.item() / sample_count]
        loss_names = ["total", "recon", "anchor", "bdsp", "smooth", "self-recon"]
        parts = [f"{loss_names[0]}: {avg_vals[0]:.3f}"]
        for i in range(1, 6):
            if avg_vals[i] > 0:
                parts.append(f"{loss_names[i]}: {avg_vals[i]:.3f}")
        parts.append(f"lr: {lr:.6f}")
        data_loader.desc = f"[val step {global_iter}] " + "  ".join(parts)

    if save_images and save_count >= max_save_images:
        print(f"\n[visualization] Reached max_save_images limit ({max_save_images}), "
              f"stopped saving. Total val samples: {step + 1}")

    if sample_count == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)

    avg_psnr = accu_psnr / psnr_count if psnr_count > 0 else None
    return (accu_total_loss.item() / sample_count, accu_recon_loss.item() / sample_count,
            accu_anchor_loss.item() / sample_count, accu_bdsp_loss.item() / sample_count,
            accu_smooth_loss.item() / sample_count, accu_self_recon_loss.item() / sample_count,
            avg_psnr)


def create_lr_scheduler(optimizer,
                        max_iterations: int,
                        warmup_iterations: int = 0,
                        warmup_factor: float = 1e-3):
    """基于 step 的学习率调度器（余弦衰减 + 线性 warmup）。"""
    assert max_iterations > 0
    total_steps = max(max_iterations - warmup_iterations, 1)

    def f(x):
        if warmup_iterations > 0 and x <= warmup_iterations:
            alpha = float(x) / float(warmup_iterations)
            return warmup_factor * (1.0 - alpha) + alpha

        post_step = x - warmup_iterations
        ratio = min(max(post_step / total_steps, 0.0), 1.0)
        return max((1.0 - ratio) ** 0.9, 0.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def save_pic(outputpic, path, index: str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = normalize_minmax(outputpic)
    outputpic = outputpic[:, :, ::-1]
    save_path = os.path.join(path, index + ".png")
    cv2.imwrite(save_path, outputpic)


def normalize_minmax(img, target_min=0, target_max=255):
    img = img * (target_max - target_min) + target_min
    return img.astype(np.uint8)


def tensor2numpy_R(R_tensor):
    R = R_tensor.squeeze(0).cpu().detach().numpy()
    R = np.transpose(R, [1, 2, 0])
    return R


def tensor2numpy_L(L_tensor):
    L = L_tensor.squeeze(0)
    L_3 = torch.cat([L, L, L], dim=0)
    L_3 = L_3.cpu().detach().numpy()
    L_3 = np.transpose(L_3, [1, 2, 0])
    return L_3
