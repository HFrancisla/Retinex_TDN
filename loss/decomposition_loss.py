"""
loss/decomposition_loss.py

Retinex 分解损失函数。

支持 8 种模式 = 4 种数据模式 × 2 种 L 表示：
  paired_point,           paired_pixel,
  unpaired_point,         unpaired_pixel,
  pure_low_double_point,  pure_low_double_pixel,
  pure_low_single_point,  pure_low_single_pixel

_point: 光照标量 L [B,1,1,1]，无 smooth。
_pixel: 逐像素光照 L [B,1,H,W]，带 Retinex smooth。
anchor_version=v1/v2 用于切换两套 Point/Pixel anchor 定义。
smooth_version=v1/v2/v3 对应 Raw/Current/Compromise 三套 Pixel smooth 定义。
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

_ANCHOR_VERSIONS = {'v1', 'v2'}


def _validate_anchor_version(anchor_version):
    if anchor_version not in _ANCHOR_VERSIONS:
        raise ValueError(
            f"anchor_version must be one of {sorted(_ANCHOR_VERSIONS)}, "
            f"got {anchor_version!r}"
        )


def _point_anchor_loss(L, I, anchor_version='v2'):
    """Point anchor，支持 v1 max-RGB map 与 v2 全局最大值。"""
    _validate_anchor_version(anchor_version)
    if anchor_version == 'v1':
        max_rgb_map = I.amax(dim=1, keepdim=True)
        return F.l1_loss(L.expand_as(max_rgb_map), max_rgb_map)

    target = I.amax(dim=(1, 2, 3), keepdim=True)
    prediction = L.mean(dim=(2, 3), keepdim=True)
    return F.l1_loss(prediction, target)


def _pixel_anchor_loss(L, I, anchor_version='v2'):
    """Pixel anchor，支持 v1 mean(max-RGB) 与 v2 mean(RGB)。"""
    _validate_anchor_version(anchor_version)
    if anchor_version == 'v1':
        target = I.amax(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
    else:
        target = I.mean(dim=(1, 2, 3), keepdim=True)
    prediction = L.mean(dim=(2, 3), keepdim=True)
    return F.l1_loss(prediction, target)


_SMOOTH_VERSIONS = {'v1', 'v2', 'v3'}


def _validate_smooth_version(smooth_version):
    if smooth_version not in _SMOOTH_VERSIONS:
        raise ValueError(
            f"smooth_version must be one of {sorted(_SMOOTH_VERSIONS)}, "
            f"got {smooth_version!r}"
        )


def _gray_reflectance(R, detach=False):
    gray = 0.299 * R[:, 0:1] + 0.587 * R[:, 1:2] + 0.114 * R[:, 2:3]
    return gray.detach() if detach else gray


def _retinex_smooth_v1(L, R):
    """Raw：2×2 卷积差分 + 零填充 + 3×3 平均，梯度进入 R/L。"""
    R_gray = _gray_reflectance(R, detach=False)
    kx = L.new_tensor([[0, 0], [-1, 1]]).view(1, 1, 2, 2)
    ky = kx.permute(0, 1, 3, 2)

    def gradient(t, kernel):
        return F.conv2d(t, kernel, stride=1, padding=1).abs()

    grad_l_x = gradient(L, kx)
    grad_l_y = gradient(L, ky)
    guide_x = F.avg_pool2d(gradient(R_gray, kx), kernel_size=3, stride=1, padding=1)
    guide_y = F.avg_pool2d(gradient(R_gray, ky), kernel_size=3, stride=1, padding=1)
    return torch.mean(
        grad_l_x * torch.exp(-10 * guide_x)
        + grad_l_y * torch.exp(-10 * guide_y)
    )


def _finite_difference_terms(L, R_gray, use_local_average):
    terms = []

    def local_average(gradient):
        if not use_local_average:
            return gradient
        padded = F.pad(gradient, (1, 1, 1, 1), mode='replicate')
        return F.avg_pool2d(padded, kernel_size=3, stride=1, padding=0)

    if L.shape[-1] > 1:
        grad_l_x = (L[:, :, :, 1:] - L[:, :, :, :-1]).abs()
        grad_r_x = (R_gray[:, :, :, 1:] - R_gray[:, :, :, :-1]).abs()
        terms.append((grad_l_x * torch.exp(-10 * local_average(grad_r_x))).mean())
    if L.shape[-2] > 1:
        grad_l_y = (L[:, :, 1:, :] - L[:, :, :-1, :]).abs()
        grad_r_y = (R_gray[:, :, 1:, :] - R_gray[:, :, :-1, :]).abs()
        terms.append((grad_l_y * torch.exp(-10 * local_average(grad_r_y))).mean())
    return sum(terms, L.new_zeros(()))


def _retinex_smooth_v2(L, R):
    """Current：真实相邻差分、无局部平均，R detach。"""
    return _finite_difference_terms(L, _gray_reflectance(R, detach=True), False)


def _retinex_smooth_v3(L, R):
    """Compromise：真实相邻差分、3×3 replicate 平均，R detach。"""
    return _finite_difference_terms(L, _gray_reflectance(R, detach=True), True)


def _retinex_smooth(L, R, smooth_version='v1'):
    """Retinex 平滑先验，v1/v2/v3 对应 Raw/Current/Compromise。

    L 在 R 平坦处应平滑，在 R 边缘处可跳变。
    L: [B, 1, H, W],  R: [B, 3, H, W]
    """
    _validate_smooth_version(smooth_version)
    if smooth_version == 'v1':
        return _retinex_smooth_v1(L, R)
    if smooth_version == 'v2':
        return _retinex_smooth_v2(L, R)
    return _retinex_smooth_v3(L, R)


def _anchor_loss(L, I, l_type, anchor_version='v2'):
    """根据 l_type 和 anchor_version 选择 anchor 损失。"""
    if l_type == 'point':
        return _point_anchor_loss(L, I, anchor_version)
    return _pixel_anchor_loss(L, I, anchor_version)


def _loss_output(total, components, **details):
    """仅输出已启用损失项的 raw 值和对 total 的加权贡献。"""
    output = {'total_loss': total}
    for name, (raw, weighted) in components.items():
        output[f'{name}_loss'] = raw
        output[f'{name}_weighted_loss'] = weighted
    output.update(details)
    return output


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
        smooth_version: str = 'v1',
        equal_r_weight: float = 0.1,
    ):
        super().__init__()
        _validate_smooth_version(smooth_version)
        self.l_type = l_type
        self.smooth_version = smooth_version
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
    def redecomp_l_consistency_weight(self) -> float:
        return 0.0

    def forward(
        self, R_low, R_high, L_low, L_high, I_low, I_high,
        L_redecomp_low=None, L_redecomp_high=None,
    ):
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
            self.Ismooth_loss_low = _retinex_smooth(L_low, R_low, self.smooth_version)
            self.Ismooth_loss_high = _retinex_smooth(L_high, R_high, self.smooth_version)
            loss_smooth = self.Ismooth_loss_low + self.Ismooth_loss_high
        else:
            self.Ismooth_loss_low = zero
            self.Ismooth_loss_high = zero
            loss_smooth = zero

        self.loss_Decom = (loss_recon + loss_cross
                           + self.smooth_weight * loss_smooth
                           + self.equal_r_weight * self.equal_R_loss)

        components = {}
        if self.recon_weight_high > 0 or self.recon_weight_low > 0:
            components['recon'] = (
                self.recon_loss_low + self.recon_loss_high, loss_recon
            )
        if self.cross_recon_weight_low > 0 or self.cross_recon_weight_high > 0:
            components['cross_recon'] = (
                self.recon_loss_crs_low + self.recon_loss_crs_high, loss_cross
            )
        if self.smooth_weight > 0:
            components['smooth'] = (loss_smooth, self.smooth_weight * loss_smooth)
        if self.equal_r_weight > 0:
            components['equal_r'] = (
                self.equal_R_loss, self.equal_r_weight * self.equal_R_loss
            )

        details = {}
        if self.recon_weight_high > 0 or self.recon_weight_low > 0:
            details.update(
                recon_low_loss=self.recon_loss_low,
                recon_high_loss=self.recon_loss_high,
            )
        if self.cross_recon_weight_low > 0 or self.cross_recon_weight_high > 0:
            details.update(
                cross_recon_low_loss=self.recon_loss_crs_low,
                cross_recon_high_loss=self.recon_loss_crs_high,
            )
        return _loss_output(self.loss_Decom, components, **details)


# ============================== UnpairedLoss ================================

class UnpairedLoss(nn.Module):
    """非配对分解损失。

    _point (默认): anchor 约束全局亮度，无 smooth。
    _pixel: anchor 只约束均值，带 Retinex smooth。
    """

    def __init__(self, l_type='point', recon_weight=1, anchor_weight=0.05,
                 bdsp_weight=0.05, smooth_weight=0,
                 redecomp_l_consistency_weight=0.05,
                 anchor_version='v2', smooth_version='v1'):
        super().__init__()
        _validate_anchor_version(anchor_version)
        _validate_smooth_version(smooth_version)
        self.l_type = l_type
        self.anchor_version = anchor_version
        self.smooth_version = smooth_version
        self.recon_weight = recon_weight
        self.anchor_weight = anchor_weight
        self.bdsp_weight = bdsp_weight
        self.redecomp_l_consistency_weight = redecomp_l_consistency_weight
        self.smooth_weight = smooth_weight

    @property
    def use_smooth(self):
        return self.l_type == 'pixel' and self.smooth_weight > 0

    def forward(
        self, R_low, R_high, L_low, L_high, I_low, I_high,
        L_redecomp_low, L_redecomp_high,
    ):
        zero = torch.tensor(0.0, device=R_low.device)
        L_low_3 = torch.cat((L_low, L_low, L_low), dim=1)
        L_high_3 = torch.cat((L_high, L_high, L_high), dim=1)

        self.recon_loss_low = F.l1_loss(R_low * L_low_3, I_low)
        self.recon_loss_high = F.l1_loss(R_high * L_high_3, I_high)
        loss_recon = self.recon_loss_low + self.recon_loss_high

        self.recon_loss_anchor_low = _anchor_loss(
            L_low, I_low, self.l_type, self.anchor_version
        )
        self.recon_loss_anchor_high = _anchor_loss(
            L_high, I_high, self.l_type, self.anchor_version
        )
        loss_anchor = self.recon_loss_anchor_low + self.recon_loss_anchor_high

        if self.bdsp_weight > 0:
            self.bdsp_loss = (F.l1_loss(BDSP_Face(R_low), BDSP_Face(I_low))
                              + F.l1_loss(BDSP_Face(R_high), BDSP_Face(I_high)))
        else:
            self.bdsp_loss = zero

        if self.redecomp_l_consistency_weight > 0:
            # 将第一次得到的 R 再分解；unpaired 图像没有空间对应关系，
            # 因此只比较二次 L 的每样本全局统计量。
            self.redecomp_l_consistency_loss = F.l1_loss(
                L_redecomp_low.mean(dim=(2, 3), keepdim=True),
                L_redecomp_high.mean(dim=(2, 3), keepdim=True),
            )
        else:
            self.redecomp_l_consistency_loss = zero

        if self.use_smooth:
            self.Ismooth_loss_low = _retinex_smooth(L_low, R_low, self.smooth_version)
            self.Ismooth_loss_high = _retinex_smooth(L_high, R_high, self.smooth_version)
            loss_smooth = self.Ismooth_loss_low + self.Ismooth_loss_high
        else:
            self.Ismooth_loss_low = zero
            self.Ismooth_loss_high = zero
            loss_smooth = zero

        self.loss_Decom = (
            self.recon_weight * loss_recon
            + self.anchor_weight * loss_anchor
            + self.bdsp_weight * self.bdsp_loss
            + self.redecomp_l_consistency_weight * self.redecomp_l_consistency_loss
            + self.smooth_weight * loss_smooth
        )

        components = {}
        if self.recon_weight > 0:
            components['recon'] = (loss_recon, self.recon_weight * loss_recon)
        if self.anchor_weight > 0:
            components['anchor'] = (loss_anchor, self.anchor_weight * loss_anchor)
        if self.bdsp_weight > 0:
            components['bdsp'] = (self.bdsp_loss, self.bdsp_weight * self.bdsp_loss)
        if self.smooth_weight > 0:
            components['smooth'] = (loss_smooth, self.smooth_weight * loss_smooth)
        if self.redecomp_l_consistency_weight > 0:
            components['redecomp_l_consistency'] = (
                self.redecomp_l_consistency_loss,
                self.redecomp_l_consistency_weight * self.redecomp_l_consistency_loss,
            )
        return _loss_output(self.loss_Decom, components)


# ============================== PureLowDoubleLoss ===========================

class PureLowDoubleLoss(nn.Module):
    """Pure-low 双视图自监督分解损失。

    _point (默认): anchor 约束全局亮度，无 smooth。
    _pixel: anchor 只约束均值，带 Retinex smooth。
    """

    def __init__(self, l_type='point', recon_weight=1.0, anchor_weight=0.05,
                 bdsp_weight=0.05, redecomp_l_consistency_weight=0.05,
                 reflect_weight=0.0, smooth_weight=0, anchor_version='v2',
                 smooth_version='v1'):
        super().__init__()
        _validate_anchor_version(anchor_version)
        _validate_smooth_version(smooth_version)
        self.l_type = l_type
        self.anchor_version = anchor_version
        self.smooth_version = smooth_version
        self.recon_weight = recon_weight
        self.anchor_weight = anchor_weight
        self.bdsp_weight = bdsp_weight
        self.redecomp_l_consistency_weight = redecomp_l_consistency_weight
        self.reflect_weight = reflect_weight
        self.smooth_weight = smooth_weight

    @property
    def use_smooth(self) -> bool:
        return self.l_type == 'pixel' and self.smooth_weight > 0

    def forward(
        self, R1, R2, L1, L2, I1, I2,
        L_redecomp_1, L_redecomp_2,
    ):
        zero = torch.tensor(0.0, device=R1.device)
        L1_3 = torch.cat((L1, L1, L1), dim=1)
        L2_3 = torch.cat((L2, L2, L2), dim=1)

        self.recon_loss_1 = F.l1_loss(R1 * L1_3, I1)
        self.recon_loss_2 = F.l1_loss(R2 * L2_3, I2)
        loss_recon = self.recon_loss_1 + self.recon_loss_2

        self.anchor_loss_1 = _anchor_loss(L1, I1, self.l_type, self.anchor_version)
        self.anchor_loss_2 = _anchor_loss(L2, I2, self.l_type, self.anchor_version)
        loss_anchor = self.anchor_loss_1 + self.anchor_loss_2

        if self.bdsp_weight > 0:
            self.bdsp_loss = (F.l1_loss(BDSP_Face(R1), BDSP_Face(I1))
                              + F.l1_loss(BDSP_Face(R2), BDSP_Face(I2)))
        else:
            self.bdsp_loss = zero

        if self.redecomp_l_consistency_weight > 0:
            self.redecomp_l_consistency_loss = F.l1_loss(
                L_redecomp_1, L_redecomp_2
            )
        else:
            self.redecomp_l_consistency_loss = zero

        if self.reflect_weight > 0:
            # 对称 stop-gradient，但取平均以保持与单个 L1 相同的数值尺度。
            self.reflect_loss = 0.5 * (
                F.l1_loss(R1, R2.detach()) + F.l1_loss(R2, R1.detach())
            )
        else:
            self.reflect_loss = zero

        if self.use_smooth:
            self.smooth_loss_1 = _retinex_smooth(L1, R1, self.smooth_version)
            self.smooth_loss_2 = _retinex_smooth(L2, R2, self.smooth_version)
            loss_smooth = self.smooth_loss_1 + self.smooth_loss_2
        else:
            loss_smooth = zero

        self.loss_Decom = (
            self.recon_weight * loss_recon
            + self.anchor_weight * loss_anchor
            + self.bdsp_weight * self.bdsp_loss
            + self.redecomp_l_consistency_weight * self.redecomp_l_consistency_loss
            + self.reflect_weight * self.reflect_loss
            + self.smooth_weight * loss_smooth
        )

        components = {}
        if self.recon_weight > 0:
            components['recon'] = (loss_recon, self.recon_weight * loss_recon)
        if self.anchor_weight > 0:
            components['anchor'] = (loss_anchor, self.anchor_weight * loss_anchor)
        if self.bdsp_weight > 0:
            components['bdsp'] = (self.bdsp_loss, self.bdsp_weight * self.bdsp_loss)
        if self.smooth_weight > 0:
            components['smooth'] = (loss_smooth, self.smooth_weight * loss_smooth)
        if self.redecomp_l_consistency_weight > 0:
            components['redecomp_l_consistency'] = (
                self.redecomp_l_consistency_loss,
                self.redecomp_l_consistency_weight * self.redecomp_l_consistency_loss,
            )
        if self.reflect_weight > 0:
            components['reflect'] = (
                self.reflect_loss, self.reflect_weight * self.reflect_loss
            )
        return _loss_output(self.loss_Decom, components)


# ============================== PureLowSingleLoss ===========================

class PureLowSingleLoss(nn.Module):
    """Pure-low 单视图分解损失。

    _point (默认): 最简约束（重建 + 全局锚定 + BDSP）。
    _pixel: 增加 Retinex smooth 约束逐像素 L 的空间平滑性。
    """

    def __init__(self, l_type='point', recon_weight=1.0, anchor_weight=0.05,
                 bdsp_weight=0.05, smooth_weight=0, anchor_version='v2',
                 smooth_version='v1'):
        super().__init__()
        _validate_anchor_version(anchor_version)
        _validate_smooth_version(smooth_version)
        self.l_type = l_type
        self.anchor_version = anchor_version
        self.smooth_version = smooth_version
        self.recon_weight = recon_weight
        self.anchor_weight = anchor_weight
        self.bdsp_weight = bdsp_weight
        self.smooth_weight = smooth_weight

    @property
    def use_smooth(self) -> bool:
        return self.l_type == 'pixel' and self.smooth_weight > 0

    def forward(self, R, L, I):
        zero = torch.tensor(0.0, device=R.device)
        L_3 = torch.cat((L, L, L), dim=1)

        loss_recon = F.l1_loss(R * L_3, I)

        loss_anchor = _anchor_loss(L, I, self.l_type, self.anchor_version)

        if self.bdsp_weight > 0:
            loss_bdsp = F.l1_loss(BDSP_Face(R), BDSP_Face(I))
        else:
            loss_bdsp = zero

        if self.use_smooth:
            loss_smooth = _retinex_smooth(L, R, self.smooth_version)
        else:
            loss_smooth = zero

        self.loss_Decom = (
            self.recon_weight * loss_recon
            + self.anchor_weight * loss_anchor
            + self.bdsp_weight * loss_bdsp
            + self.smooth_weight * loss_smooth
        )

        components = {}
        if self.recon_weight > 0:
            components['recon'] = (loss_recon, self.recon_weight * loss_recon)
        if self.anchor_weight > 0:
            components['anchor'] = (loss_anchor, self.anchor_weight * loss_anchor)
        if self.bdsp_weight > 0:
            components['bdsp'] = (loss_bdsp, self.bdsp_weight * loss_bdsp)
        if self.smooth_weight > 0:
            components['smooth'] = (loss_smooth, self.smooth_weight * loss_smooth)
        return _loss_output(self.loss_Decom, components)
