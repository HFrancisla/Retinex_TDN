"""
utils.py

训练与评估辅助函数。

包含数据路径读取、单 epoch 训练/验证、学习率 warmup 调度器、
结果可视化保存等工具函数。
"""

import os
import sys

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
    if pred.shape != target.shape:
        raise ValueError(f"PSNR shape mismatch: pred={pred.shape}, target={target.shape}")
    mse = (pred - target).square().flatten(1).mean(dim=1)
    psnr = torch.where(
        mse == 0,
        torch.full_like(mse, float('inf')),
        10.0 * torch.log10((max_val ** 2) / mse),
    )
    return psnr.mean().item()


def calculate_l1(pred, target):
    """Return mean absolute error per image, averaged across the batch."""
    if pred.shape != target.shape:
        raise ValueError(f"L1 shape mismatch: pred={pred.shape}, target={target.shape}")
    return (pred - target).abs().flatten(1).mean(dim=1).mean().item()




def load_config(config_path: str) -> dict:
    """
    加载 YAML 配置文件。

    Args:
        config_path: YAML 配置文件路径

    Returns:
        dict: 配置字典
    """
    import yaml

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config file does not exist: {config_path}")

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
        'smooth_weight', 'smooth_version', 'equal_r_weight',
    },
    'unpaired_point': {
        'recon_weight', 'anchor_weight', 'bdsp_weight',
        'redecomp_l_consistency_weight', 'anchor_version',
    },
    'unpaired_pixel': {
        'recon_weight', 'anchor_weight', 'bdsp_weight',
        'smooth_weight', 'smooth_version', 'redecomp_l_consistency_weight',
        'anchor_version',
    },
    'pure_low_double_point': {
        'recon_weight', 'anchor_weight', 'bdsp_weight',
        'redecomp_l_consistency_weight', 'reflect_weight', 'anchor_version',
    },
    'pure_low_double_pixel': {
        'recon_weight', 'anchor_weight', 'bdsp_weight',
        'smooth_weight', 'smooth_version', 'redecomp_l_consistency_weight',
        'reflect_weight', 'anchor_version',
    },
    'pure_low_single_point': {
        'recon_weight', 'anchor_weight', 'bdsp_weight', 'anchor_version',
    },
    'pure_low_single_pixel': {
        'recon_weight', 'anchor_weight', 'bdsp_weight',
        'smooth_weight', 'smooth_version', 'anchor_version',
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
    含 anchor 的模式必须显式指定 anchor_version: v1 或 v2。
    所有 Pixel 模式必须显式指定 smooth_version: v1、v2 或 v3。
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
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    train_root = os.path.join(root, "train")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"train root does not exist: {train_root}")

    val_root = os.path.join(root, "val")
    if not os.path.exists(val_root):
        test_root = os.path.join(root, "test")
        if os.path.exists(test_root):
            print(f"[read_data] 'val' not found, fallback to 'test'")
            val_root = test_root
        else:
            raise FileNotFoundError(
                f"Neither 'val' nor 'test' directory exists in {root}."
            )

    train_images_low_path = []
    train_images_high_path = []
    val_images_low_path = []
    val_images_high_path = []

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    train_high_root = os.path.join(train_root, "high")
    train_low_root = os.path.join(train_root, "low")

    val_high_root = os.path.join(val_root, "high")
    val_low_root = os.path.join(val_root, "low")

    def _images(folder):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"image directory does not exist: {folder}")
        return sorted(
            os.path.join(folder, name) for name in os.listdir(folder)
            if os.path.splitext(name)[1].lower() in supported
        )

    train_low_path = _images(train_low_root)
    train_high_path = _images(train_high_root)
    val_low_path = _images(val_low_root)
    val_high_path = _images(val_high_root)

    if mode == "paired":
        def _pair_by_stem(low_paths, high_paths, split):
            def _index(paths, domain):
                result = {}
                for path in paths:
                    stem = os.path.splitext(os.path.basename(path))[0]
                    if stem in result:
                        raise ValueError(f"duplicate {domain} image stem '{stem}' in {split}")
                    result[stem] = path
                return result

            low_by_stem = _index(low_paths, 'low')
            high_by_stem = _index(high_paths, 'high')
            if low_by_stem.keys() != high_by_stem.keys():
                missing_high = sorted(low_by_stem.keys() - high_by_stem.keys())
                missing_low = sorted(high_by_stem.keys() - low_by_stem.keys())
                raise ValueError(
                    f"paired filename mismatch in {split}: "
                    f"missing high={missing_high[:10]}, missing low={missing_low[:10]}"
                )
            stems = sorted(low_by_stem)
            return ([low_by_stem[s] for s in stems], [high_by_stem[s] for s in stems])

        train_low_path, train_high_path = _pair_by_stem(train_low_path, train_high_path, 'train')
        val_low_path, val_high_path = _pair_by_stem(val_low_path, val_high_path, 'val')
        print("image pair check finish")
    else:
        if mode in ("pure_low_single", "pure_low_double"):
            print("pure_low mode: low-only training, train({}), val({})".format(len(train_low_path), len(val_low_path)))
        else:
            print("unpaired mode: low({}) and high({}) loaded independently".format(len(train_low_path), len(train_high_path)))
    if not train_low_path or not val_low_path:
        raise ValueError(
            f"dataset must contain low images in both train and validation: "
            f"train={len(train_low_path)}, val={len(val_low_path)}"
        )
    if mode in ('paired', 'unpaired') and (not train_high_path or not val_high_path):
        raise ValueError(
            f"mode={mode} requires high images in both train and validation: "
            f"train={len(train_high_path)}, val={len(val_high_path)}"
        )

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





def read_pure_low_data(root: str, include_val_high_ref: bool = False):
    """读取 pure_low 数据集。

    训练始终只返回 low 图像；当 ``include_val_high_ref`` 为 true 且验证集存在
    同名 high 图像时，额外返回验证用 high reference 路径。
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    train_root = os.path.join(root, "train")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"train root does not exist: {train_root}")

    val_root = os.path.join(root, "val")
    if not os.path.exists(val_root):
        test_root = os.path.join(root, "test")
        if os.path.exists(test_root):
            print(f"[read_pure_low_data] 'val' not found, fallback to 'test'")
            val_root = test_root
        else:
            raise FileNotFoundError(
                f"Neither 'val' nor 'test' directory exists in {root}."
            )

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

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
         if os.path.splitext(i)[-1].lower() in supported]
    )
    val_low_path = sorted(
        [os.path.join(val_low_root, i) for i in os.listdir(val_low_root)
         if os.path.splitext(i)[-1].lower() in supported]
    )

    val_high_ref_path = []
    if include_val_high_ref:
        val_high_root = os.path.join(val_root, "high")
        if os.path.isdir(val_high_root):
            high_paths = sorted(
                os.path.join(val_high_root, i) for i in os.listdir(val_high_root)
                if os.path.splitext(i)[-1].lower() in supported
            )
            low_by_stem = {
                os.path.splitext(os.path.basename(path))[0]: path
                for path in val_low_path
            }
            high_by_stem = {
                os.path.splitext(os.path.basename(path))[0]: path
                for path in high_paths
            }
            if low_by_stem.keys() == high_by_stem.keys():
                val_low_path = [low_by_stem[stem] for stem in sorted(low_by_stem)]
                val_high_ref_path = [high_by_stem[stem] for stem in sorted(low_by_stem)]
            elif high_paths:
                missing_high = sorted(low_by_stem.keys() - high_by_stem.keys())
                missing_low = sorted(high_by_stem.keys() - low_by_stem.keys())
                raise ValueError(
                    "pure-low validation high reference filenames do not match low: "
                    f"missing high={missing_high[:10]}, missing low={missing_low[:10]}"
                )

    if not train_low_path or not val_low_path:
        raise ValueError(
            f"pure-low dataset must contain images in both train and validation: "
            f"train={len(train_low_path)}, val={len(val_low_path)}"
        )
    print("pure_low loader: train({}), val({})".format(len(train_low_path), len(val_low_path)))
    if include_val_high_ref:
        print("pure_low validation high refs: {}".format(len(val_high_ref_path)))
        return list(train_low_path), list(val_low_path), list(val_high_ref_path)
    return list(train_low_path), list(val_low_path)
def _forward_loss(model, loss_function, data, device):
    """统一四种数据模式的前向和损失调用。"""
    non_blocking = device.type == 'cuda'
    if isinstance(loss_function, PureLowSingleLoss):
        if isinstance(data, (tuple, list)):
            I_low, I_high = data
            I_high = I_high.to(device, non_blocking=non_blocking)
        else:
            I_low, I_high = data, None
        I_low = I_low.to(device, non_blocking=non_blocking)
        R_low, L_low = model(I_low)
        return (
            loss_function(R_low, L_low, I_low),
            I_low, I_high, R_low, L_low, None, None,
        )

    I_low, I_high = data
    I_low = I_low.to(device, non_blocking=non_blocking)
    I_high = I_high.to(device, non_blocking=non_blocking)
    R_low, L_low = model(I_low)
    R_high, L_high = model(I_high)

    if loss_function.redecomp_l_consistency_weight > 0:
        _, L_redecomp_low = model(R_low)
        _, L_redecomp_high = model(R_high)
    else:
        L_redecomp_low = None
        L_redecomp_high = None

    output = loss_function(
        R_low, R_high, L_low, L_high, I_low, I_high,
        L_redecomp_low, L_redecomp_high,
    )
    return output, I_low, I_high, R_low, L_low, R_high, L_high


def train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch, loss_cfg=None):
    """兼容旧调用的单 epoch 训练；统计按样本数加权。"""
    loss_function = _build_loss_function(loss_cfg).to(device)
    totals = {}
    sample_count = 0
    for data in tqdm(data_loader, file=sys.stdout, desc=f'[train epoch {epoch}]'):
        metrics = train_step(model, optimizer, loss_function, data, device, lr_scheduler)
        batch_size = int(metrics.pop('_batch_size'))
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        sample_count += batch_size
    averages = {key: value / sample_count for key, value in totals.items()} if sample_count else {}
    averages['lr'] = optimizer.param_groups[0]['lr']
    return averages


def train_step(model, optimizer, loss_function, data, device, lr_scheduler):
    """单步训练：在参数更新前拦截非有限 loss/gradient。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output, I_low, _, _, _, _, _ = _forward_loss(model, loss_function, data, device)
    loss = output['total_loss']

    if not torch.isfinite(loss):
        raise FloatingPointError(f'non-finite loss before optimizer step: {loss.detach().item()}')
    loss.backward()

    bad_gradient = next(
        (name for name, param in model.named_parameters()
         if param.grad is not None and not torch.isfinite(param.grad).all()),
        None,
    )
    if bad_gradient is not None:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(f'non-finite gradient before optimizer step: {bad_gradient}')

    optimizer.step()
    lr_scheduler.step()

    metrics = {key: value.detach().item() for key, value in output.items()}
    metrics['_batch_size'] = I_low.shape[0]
    return metrics


@torch.no_grad()
def evaluate(model, data_loader, device, lr, filefold_path,
             loss_function=None, loss_cfg=None, save_images=False, global_iter=0,
             max_save_images=250):

    if loss_function is None:
        loss_function = _build_loss_function(loss_cfg)

    model.eval()

    metric_sums = {}
    accu_psnr = 0.0
    psnr_count = 0
    sample_count = 0

    loss_function = loss_function.to(device)

    if save_images:
        evalfold_path = os.path.join(filefold_path, str(global_iter))
        os.makedirs(evalfold_path, exist_ok=True)

    save_count = 0  # 已保存图像计数，达到 max_save_images 后停止保存

    # 动态进度条仅在真实终端中启用。日志采集器通常无法解释 \r，
    # 会把每次刷新展开成新内容；简短描述也可避免窄终端自动折行。
    data_loader = tqdm(
        data_loader,
        desc=f"[val step {global_iter}]",
        file=sys.stdout,
        disable=not sys.stdout.isatty(),
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
    )
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            output, I_low, I_high, R_low, L_low, R_high, L_high = _forward_loss(
                model, loss_function, data, device
            )

            if save_images and save_count < max_save_images:
                remaining = max_save_images - save_count
                for batch_index in range(min(I_low.shape[0], remaining)):
                    image_index = sample_count + batch_index
                    illumination = L_low[batch_index]
                    if illumination.shape[-2:] != R_low.shape[-2:]:
                        illumination = illumination.expand(
                            1, R_low.shape[-2], R_low.shape[-1]
                        )
                    save_pic(tensor2numpy_R(R_low[batch_index]), evalfold_path,
                             f"{image_index}_R_low")
                    save_pic(tensor2numpy_L(illumination), evalfold_path,
                             f"{image_index}_L_low")

                    # 只有 paired 验证集中的 high 与 low 是同一场景，才保存
                    # high 分解用于逐样本对照。unpaired/pure-low 模式不保存 high。
                    if isinstance(loss_function, PairedLoss):
                        high_illumination = L_high[batch_index]
                        if high_illumination.shape[-2:] != R_high.shape[-2:]:
                            high_illumination = high_illumination.expand(
                                1, R_high.shape[-2], R_high.shape[-1]
                            )
                        save_pic(tensor2numpy_R(R_high[batch_index]), evalfold_path,
                                 f"{image_index}_R_high")
                        save_pic(tensor2numpy_L(high_illumination), evalfold_path,
                                 f"{image_index}_L_high")
                    save_count += 1

            # paired loss 专用：记录 R_low vs R_high 之间的反射分量一致性。
            if isinstance(loss_function, PairedLoss) and R_high is not None:
                for batch_index in range(I_low.shape[0]):
                    r_low = R_low[batch_index:batch_index + 1].clamp(0, 1)
                    r_high = R_high[batch_index:batch_index + 1].clamp(0, 1)

                    consistency_psnr = calculate_psnr(
                        r_low,
                        r_high,
                    )
                    metric_sums['r_consistency_psnr'] = (
                        metric_sums.get('r_consistency_psnr', 0.0) + consistency_psnr
                    )

            # 配对数据集可用的实用参考指标。pure-low 训练也可以只在验证阶段携带
            # I_high_ref，用于选择 checkpoint，但不进入训练 loss。
            if I_high is not None:
                for batch_index in range(I_low.shape[0]):
                    r_low = R_low[batch_index:batch_index + 1].clamp(0, 1)
                    high_ref = I_high[batch_index:batch_index + 1].clamp(0, 1)
                    psnr_val = calculate_psnr(r_low, high_ref)
                    accu_psnr += psnr_val
                    psnr_count += 1

                    metric_sums['r_low_highref_psnr'] = (
                        metric_sums.get('r_low_highref_psnr', 0.0)
                        + psnr_val
                    )
                    metric_sums['r_low_highref_l1'] = (
                        metric_sums.get('r_low_highref_l1', 0.0)
                        + calculate_l1(r_low, high_ref)
                    )
                    metric_sums['r_low_highref_mean_ratio'] = (
                        metric_sums.get('r_low_highref_mean_ratio', 0.0)
                        + (r_low.mean() / high_ref.mean().clamp_min(1e-8)).item()
                    )
                    metric_sums['r_low_highref_overbright_010'] = (
                        metric_sums.get('r_low_highref_overbright_010', 0.0)
                        + (r_low > high_ref + 0.1).float().mean().item()
                    )

            batch_size = I_low.shape[0]

            # loss 函数返回 batch 内均值，因此按 batch 样本数加权，确保最后一个
            # 不足 batch_size 的 batch 不会与完整 batch 获得相同权重。
            for key, value in output.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value.detach().item() * batch_size
            sample_count += batch_size

    if save_images and save_count >= max_save_images:
        print(f"\n[visualization] Reached max_save_images limit ({max_save_images}), "
              f"stopped saving. Total val samples: {sample_count}")

    if sample_count == 0:
        return ({}, None)

    avg_psnr = accu_psnr / psnr_count if psnr_count > 0 else None
    averages = {key: value / sample_count for key, value in metric_sums.items()}
    return averages, avg_psnr


def create_lr_scheduler(optimizer,
                        max_iterations: int,
                        warmup_iterations: int = 0,
                        warmup_factor: float = 1e-3):
    """基于 step 的学习率调度器（0.9 次多项式衰减 + 线性 warmup）。"""
    if max_iterations <= 0:
        raise ValueError(f'max_iterations must be > 0, got {max_iterations}')
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
    if not cv2.imwrite(save_path, outputpic):
        raise OSError(f"failed to save image: {save_path}")


def normalize_minmax(img, target_min=0, target_max=255):
    img = img * (target_max - target_min) + target_min
    return img.astype(np.uint8)


def tensor2numpy_R(R_tensor):
    if R_tensor.ndim == 4:
        if R_tensor.shape[0] != 1:
            raise ValueError(f"tensor2numpy_R expects one image, got {R_tensor.shape}")
        R_tensor = R_tensor[0]
    if R_tensor.ndim != 3 or R_tensor.shape[0] != 3:
        raise ValueError(f"tensor2numpy_R expects [3,H,W], got {R_tensor.shape}")
    R = R_tensor.cpu().detach().numpy()
    R = np.transpose(R, [1, 2, 0])
    return R


def tensor2numpy_L(L_tensor):
    if L_tensor.ndim == 4:
        if L_tensor.shape[0] != 1:
            raise ValueError(f"tensor2numpy_L expects one image, got {L_tensor.shape}")
        L_tensor = L_tensor[0]
    if L_tensor.ndim != 3 or L_tensor.shape[0] != 1:
        raise ValueError(f"tensor2numpy_L expects [1,H,W], got {L_tensor.shape}")
    L_3 = L_tensor.repeat(3, 1, 1)
    L_3 = L_3.cpu().detach().numpy()
    L_3 = np.transpose(L_3, [1, 2, 0])
    return L_3
