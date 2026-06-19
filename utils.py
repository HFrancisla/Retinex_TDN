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

from loss import PairedLoss, UnpairedLoss


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


def _build_loss_function(loss_cfg):
    """根据 loss_cfg 构建损失模块。

    loss_cfg 中的 'mode' 字段决定使用哪套损失：
    - 'paired' -> PairedLoss
    - 'unpaired' -> UnpairedLoss
    其余字段透传给对应损失类的 __init__。
    """
    cfg = dict(loss_cfg or {})
    mode = cfg.pop('mode', 'unpaired')
    if mode == 'paired':
        return PairedLoss(**cfg)
    return UnpairedLoss(**cfg)


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

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    lr = optimizer.param_groups[0]["lr"]
    for step, data in enumerate(data_loader):
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


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, lr, filefold_path, loss_cfg=None):
    loss_function = _build_loss_function(loss_cfg)

    model.eval()

    accu_total_loss = torch.zeros(1).to(device)
    accu_recon_loss = torch.zeros(1).to(device)
    accu_anchor_loss = torch.zeros(1).to(device)
    accu_bdsp_loss = torch.zeros(1).to(device)
    accu_smooth_loss = torch.zeros(1).to(device)
    accu_self_recon_loss = torch.zeros(1).to(device)
    save_epoch = 20

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
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

        if epoch % save_epoch == 0:
            R_low_img = tensor2numpy_R(R_low)
            R_high_img = tensor2numpy_R(R_high)
            L_low_img = tensor2numpy_L(L_low)
            L_high_img = tensor2numpy_L(L_high)
            save_pic(R_low_img, evalfold_path, str(step) + "_R_low")
            save_pic(R_high_img, evalfold_path, str(step) + "_R_high")
            save_pic(L_low_img, evalfold_path, str(step) + "_L_low")
            save_pic(L_high_img, evalfold_path, str(step) + "_L_high")

        loss, loss_recon, loss_anchor, loss_bdsp, loss_smooth, loss_self_recon = \
            loss_function(R_low, R_high, L_low, L_high, I_low, I_high, L_R_low, L_R_high)

        accu_total_loss += loss
        accu_recon_loss += loss_recon
        accu_anchor_loss += loss_anchor
        accu_bdsp_loss += loss_bdsp
        accu_smooth_loss += loss_smooth
        accu_self_recon_loss += loss_self_recon

        n = step + 1
        data_loader.desc = (
            "[val epoch {}] total: {:.3f}  recon: {:.3f}  anchor: {:.3f}  "
            "bdsp: {:.3f}  smooth: {:.3f}  self-recon: {:.3f}  lr: {:.6f}"
        ).format(
            epoch,
            accu_total_loss.item() / n, accu_recon_loss.item() / n,
            accu_anchor_loss.item() / n, accu_bdsp_loss.item() / n,
            accu_smooth_loss.item() / n, accu_self_recon_loss.item() / n, lr
        )

    step_count = max(step + 1, 1) if data_loader.iterable else 0
    if step_count == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return (accu_total_loss.item() / step_count, accu_recon_loss.item() / step_count,
            accu_anchor_loss.item() / step_count, accu_bdsp_loss.item() / step_count,
            accu_smooth_loss.item() / step_count, accu_self_recon_loss.item() / step_count)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    warmup_steps = warmup_epochs * num_step
    total_steps = max((epochs - warmup_epochs) * num_step, 1)

    def f(x):
        if warmup and x <= warmup_steps:
            alpha = float(x) / float(warmup_steps)
            return warmup_factor * (1.0 - alpha) + alpha

        post_step = x - warmup_steps
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
