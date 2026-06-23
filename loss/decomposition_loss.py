"""
loss/decomposition_loss.py

Retinex 分解损失函数。

支持 8 种模式 = 4 种数据模式 × 2 种 L 表示：
  paired_point,           paired_pixel,
  unpaired_point,         unpaired_pixel,
  pure_low_double_point,  pure_low_double_pixel,
  pure_low_single_point,  pure_low_single_pixel

_point: 光照标量 L [B,1,1,1]，anchor 约束全局亮度，无 smooth。
_pixel: 逐像素光照 L [B,1,H,W]，anchor 只约束均值亮度，带 Retinex smooth。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .bsdp import BDSP_Face


# ============================== 辅助函数 ====================================

Sobel = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])
Robert = np.array([[0, 0],
                   [-1, 1]])
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)


def gradient(maps, direction, device=None, kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 1, 0, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device or maps.device)
    gradient_orig = F.conv2d(maps, weight=kernel, padding=0).abs()
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = (gradient_orig - grad_min) / (grad_max - grad_min + 1e-4)
    return grad_norm


def gradient_no_abs(maps, direction, device=None, kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 1, 0, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device or maps.device)
    return F.conv2d(maps, weight=kernel, padding=0)


# ========================= point / pixel 共用辅助 ============================

def _point_anchor_loss(L, I):
    """Point anchor: L1(max(I), L) — 标量 L 空间均值自然生效。"""
    max_val = torch.max(I, dim=1, keepdim=True)[0]
    return F.l1_loss(max_val, L.expand_as(max_val))


def _pixel_anchor_loss(L, I):
    """Pixel anchor: 只约束均值亮度，避免逐像素 shortcut。"""
    max_val = torch.max(I, dim=1, keepdim=True)[0]
    return F.l1_loss(
        max_val.mean(dim=[2, 3], keepdim=True),
        L.mean(dim=[2, 3], keepdim=True),
    )


def _retinex_smooth(L, R):
    """Retinex 平滑先验: |∇L| * exp(-10 * avg|∇R_gray|)。

    L 在 R 平坦处应平滑，在 R 边缘处可跳变。
    L: [B, 1, H, W],  R: [B, 3, H, W]
    """
    R_gray = 0.299 * R[:, 0:1] + 0.587 * R[:, 1:2] + 0.114 * R[:, 2:3]
    dev = L.device
    kx = torch.FloatTensor([[0, 0], [-1, 1]]).view(1, 1, 2, 2).to(dev)
    ky = kx.permute(0, 1, 3, 2)

    def _g(t, d):
        return F.conv2d(t, kx if d == 'x' else ky, stride=1, padding=1).abs()

    def _ag(t, d):
        return F.avg_pool2d(_g(t, d), kernel_size=3, stride=1, padding=1)

    return torch.mean(
        _g(L, 'x') * torch.exp(-10 * _ag(R_gray, 'x'))
        + _g(L, 'y') * torch.exp(-10 * _ag(R_gray, 'y'))
    )


def _anchor_loss(L, I, l_type):
    """根据 l_type 选择 anchor 损失。"""
    if l_type == 'point':
        return _point_anchor_loss(L, I)
    return _pixel_anchor_loss(L, I)


# ============================== PairedLoss ==================================

class PairedLoss(nn.Module):
    """成对分解损失（有监督）。

    _pixel: 自重建 + 交叉重建 + R 一致性 + Retinex smooth。
    _point: 去掉 smooth（标量 L 无意义）。
    """

    def __init__(
        self,
        l_type: str = 'pixel',
        recon_weight_high: float = 1.0,
        recon_weight_low: float = 0.3,
        cross_recon_weight_low: float = 0.001,
        cross_recon_weight_high: float = 0.001,
        smooth_weight: float = 0.1,
        equal_r_weight: float = 0.1,
    ):
        super().__init__()
        self.l_type = l_type
        self.recon_weight_high = recon_weight_high
        self.recon_weight_low = recon_weight_low
        self.cross_recon_weight_low = cross_recon_weight_low
        self.cross_recon_weight_high = cross_recon_weight_high
        self.equal_r_weight = equal_r_weight
        # point 模式下 smooth 无意义，强制为 0
        self.smooth_weight = 0 if l_type == 'point' else smooth_weight

    @property
    def use_smooth(self) -> bool:
        return self.smooth_weight > 0

    @property
    def self_recon_weight(self) -> float:
        return 0.0

    def forward(self, R_low, R_high, L_low, L_high, I_low, I_high, L_R_low=None, L_R_high=None):
        zero = torch.tensor(0.0, device=R_low.device)
        L_low_3 = torch.cat((L_low, L_low, L_low), dim=1)
        L_high_3 = torch.cat((L_high, L_high, L_high), dim=1)

        self.recon_loss_low = F.l1_loss(R_low * L_low_3, I_low)
        self.recon_loss_high = F.l1_loss(R_high * L_high_3, I_high)
        loss_recon = self.recon_weight_high * self.recon_loss_high + self.recon_weight_low * self.recon_loss_low

        self.recon_loss_crs_low = F.l1_loss(R_high * L_low_3, I_low)
        self.recon_loss_crs_high = F.l1_loss(R_low * L_high_3, I_high)
        loss_cross = (self.cross_recon_weight_low * self.recon_loss_crs_low
                      + self.cross_recon_weight_high * self.recon_loss_crs_high)

        if self.equal_r_weight > 0:
            self.equal_R_loss = F.l1_loss(R_low, R_high.detach())
        else:
            self.equal_R_loss = zero

        if self.use_smooth:
            self.Ismooth_loss_low = _retinex_smooth(L_low, R_low)
            self.Ismooth_loss_high = _retinex_smooth(L_high, R_high)
            loss_smooth = self.Ismooth_loss_low + self.Ismooth_loss_high
        else:
            self.Ismooth_loss_low = zero
            self.Ismooth_loss_high = zero
            loss_smooth = zero

        self.loss_Decom = (loss_recon + loss_cross
                           + self.smooth_weight * loss_smooth
                           + self.equal_r_weight * self.equal_R_loss)

        return (
            self.loss_Decom,
            self.recon_loss_low + self.recon_loss_high,
            zero,   # anchor placeholder
            zero,   # bdsp placeholder
            loss_smooth,
            zero,   # self_recon placeholder
        )


# ============================== UnpairedLoss ================================

class UnpairedLoss(nn.Module):
    """非配对分解损失。

    _point (默认): anchor 约束全局亮度，无 smooth。
    _pixel: anchor 只约束均值，带 Retinex smooth。
    """

    def __init__(self, l_type='point', recon_weight=1, anchor_weight=0.05,
                 bdsp_weight=0.05, smooth_weight=0, self_recon_weight=0.05):
        super().__init__()
        self.l_type = l_type
        self.recon_weight = recon_weight
        self.anchor_weight = anchor_weight
        self.bdsp_weight = bdsp_weight
        self.self_recon_weight = self_recon_weight
        # pixel 模式默认开启 smooth（用户未显式设置时）
        if l_type == 'pixel' and smooth_weight == 0:
            self.smooth_weight = 0.1
        else:
            self.smooth_weight = smooth_weight

    @property
    def use_smooth(self):
        return self.l_type == 'pixel' and self.smooth_weight > 0

    def forward(self, R_low, R_high, L_low, L_high, I_low, I_high, L_R_low, L_R_high):
        zero = torch.tensor(0.0, device=R_low.device)
        L_low_3 = torch.cat((L_low, L_low, L_low), dim=1)
        L_high_3 = torch.cat((L_high, L_high, L_high), dim=1)

        self.recon_loss_low = F.l1_loss(R_low * L_low_3, I_low)
        self.recon_loss_high = F.l1_loss(R_high * L_high_3, I_high)
        loss_recon = self.recon_loss_low + self.recon_loss_high

        self.recon_loss_anchor_low = _anchor_loss(L_low, I_low, self.l_type)
        self.recon_loss_anchor_high = _anchor_loss(L_high, I_high, self.l_type)
        loss_anchor = self.recon_loss_anchor_low + self.recon_loss_anchor_high

        if self.bdsp_weight > 0:
            self.bdsp_loss = (F.l1_loss(BDSP_Face(R_low), BDSP_Face(I_low))
                              + F.l1_loss(BDSP_Face(R_high), BDSP_Face(I_high)))
        else:
            self.bdsp_loss = zero

        if self.self_recon_weight > 0:
            self.self_recon_loss = F.l1_loss(L_R_low, L_R_high)
        else:
            self.self_recon_loss = zero

        if self.use_smooth:
            self.Ismooth_loss_low = _retinex_smooth(L_low, R_low)
            self.Ismooth_loss_high = _retinex_smooth(L_high, R_high)
            loss_smooth = self.Ismooth_loss_low + self.Ismooth_loss_high
        else:
            self.Ismooth_loss_low = zero
            self.Ismooth_loss_high = zero
            loss_smooth = zero

        self.loss_Decom = (
            self.recon_weight * loss_recon
            + self.anchor_weight * loss_anchor
            + self.bdsp_weight * self.bdsp_loss
            + self.self_recon_weight * self.self_recon_loss
            + self.smooth_weight * loss_smooth
        )

        return (self.loss_Decom, loss_recon, loss_anchor,
                self.bdsp_loss, loss_smooth, self.self_recon_loss)


# ============================== PureLowDoubleLoss ===========================

class PureLowDoubleLoss(nn.Module):
    """Pure-low 双视图自监督分解损失。

    _point (默认): anchor 约束全局亮度，无 smooth。
    _pixel: anchor 只约束均值，带 Retinex smooth。
    """

    def __init__(self, l_type='point', recon_weight=1.0, anchor_weight=0.05,
                 bdsp_weight=0.05, self_recon_weight=0.05, reflect_weight=0.0,
                 smooth_weight=0):
        super().__init__()
        self.l_type = l_type
        self.recon_weight = recon_weight
        self.anchor_weight = anchor_weight
        self.bdsp_weight = bdsp_weight
        self.self_recon_weight = self_recon_weight
        self.reflect_weight = reflect_weight
        if l_type == 'pixel' and smooth_weight == 0:
            self.smooth_weight = 0.1
        else:
            self.smooth_weight = smooth_weight

    @property
    def use_smooth(self) -> bool:
        return self.l_type == 'pixel' and self.smooth_weight > 0

    def forward(self, R1, R2, L1, L2, I1, I2, LR1, LR2):
        zero = torch.tensor(0.0, device=R1.device)
        L1_3 = torch.cat((L1, L1, L1), dim=1)
        L2_3 = torch.cat((L2, L2, L2), dim=1)

        self.recon_loss_1 = F.l1_loss(R1 * L1_3, I1)
        self.recon_loss_2 = F.l1_loss(R2 * L2_3, I2)
        loss_recon = self.recon_loss_1 + self.recon_loss_2

        self.anchor_loss_1 = _anchor_loss(L1, I1, self.l_type)
        self.anchor_loss_2 = _anchor_loss(L2, I2, self.l_type)
        loss_anchor = self.anchor_loss_1 + self.anchor_loss_2

        if self.bdsp_weight > 0:
            self.bdsp_loss = (F.l1_loss(BDSP_Face(R1), BDSP_Face(I1))
                              + F.l1_loss(BDSP_Face(R2), BDSP_Face(I2)))
        else:
            self.bdsp_loss = zero

        if self.self_recon_weight > 0:
            self.self_recon_loss = F.l1_loss(LR1, LR2)
        else:
            self.self_recon_loss = zero

        if self.reflect_weight > 0:
            self.reflect_loss = F.l1_loss(R1, R2.detach()) + F.l1_loss(R2, R1.detach())
        else:
            self.reflect_loss = zero

        if self.use_smooth:
            self.smooth_loss_1 = _retinex_smooth(L1, R1)
            self.smooth_loss_2 = _retinex_smooth(L2, R2)
            loss_smooth = self.smooth_loss_1 + self.smooth_loss_2
        else:
            loss_smooth = zero

        self.loss_Decom = (
            self.recon_weight * loss_recon
            + self.anchor_weight * loss_anchor
            + self.bdsp_weight * self.bdsp_loss
            + self.self_recon_weight * self.self_recon_loss
            + self.reflect_weight * self.reflect_loss
            + self.smooth_weight * loss_smooth
        )

        return (self.loss_Decom, loss_recon, loss_anchor,
                self.bdsp_loss, loss_smooth, self.self_recon_loss)


# ============================== PureLowSingleLoss ===========================

class PureLowSingleLoss(nn.Module):
    """Pure-low 单视图分解损失。

    _point (默认): 最简约束（重建 + 全局锚定 + BDSP）。
    _pixel: 增加 Retinex smooth 约束逐像素 L 的空间平滑性。
    """

    def __init__(self, l_type='point', recon_weight=1.0, anchor_weight=0.05,
                 bdsp_weight=0.05, smooth_weight=0):
        super().__init__()
        self.l_type = l_type
        self.recon_weight = recon_weight
        self.anchor_weight = anchor_weight
        self.bdsp_weight = bdsp_weight
        if l_type == 'pixel' and smooth_weight == 0:
            self.smooth_weight = 0.1
        else:
            self.smooth_weight = smooth_weight

    @property
    def use_smooth(self) -> bool:
        return self.l_type == 'pixel' and self.smooth_weight > 0

    def forward(self, R, L, I):
        zero = torch.tensor(0.0, device=R.device)
        L_3 = torch.cat((L, L, L), dim=1)

        loss_recon = F.l1_loss(R * L_3, I)

        loss_anchor = _anchor_loss(L, I, self.l_type)

        if self.bdsp_weight > 0:
            loss_bdsp = F.l1_loss(BDSP_Face(R), BDSP_Face(I))
        else:
            loss_bdsp = zero

        if self.use_smooth:
            loss_smooth = _retinex_smooth(L, R)
        else:
            loss_smooth = zero

        self.loss_Decom = (
            self.recon_weight * loss_recon
            + self.anchor_weight * loss_anchor
            + self.bdsp_weight * loss_bdsp
            + self.smooth_weight * loss_smooth
        )

        return (self.loss_Decom, loss_recon, loss_anchor, loss_bdsp, loss_smooth, zero)
