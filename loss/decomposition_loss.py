"""
loss/decomposition_loss.py

Retinex 分解损失函数。

包含两种模式：
- paired: 成对/有监督分解损失（参考 Diff-TDN 风格），支持自重建与交叉重建、光照平滑、反射率一致性等项。
- unpaired: 非配对/无监督分解损失（当前项目默认风格），支持重建、光照锚定、BDSP、光照平滑、自重构约束等项。

当 config 中 `data.mode=paired` 时默认走 paired 损失，`data.mode=unpaired` 时默认走 unpaired 损失；
也可以通过显式设置 `loss.mode` 强制覆盖（例如 `loss.mode: paired` 或 `loss.mode: unpaired`）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .bsdp import BDSP_Face


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


class PairedLoss(nn.Module):
    """Diff-Retinex 风格的成对分解损失（可配置权重）。

    返回值与现有训练/验证日志兼容：
    (loss, recon_loss, anchor_loss_placeholder, bdsp_loss_placeholder, smooth_loss, self_recon_placeholder)
    其中 anchor/bdsp/self_recon 占位为 0，保持日志接口稳定。
    """

    def __init__(
        self,
        recon_weight_high: float = 1.0,
        recon_weight_low: float = 0.3,
        cross_recon_weight_low: float = 0.001,
        cross_recon_weight_high: float = 0.001,
        smooth_weight: float = 0.1,
        equal_r_weight: float = 0.1,
    ):
        super().__init__()
        self.recon_weight_high = recon_weight_high
        self.recon_weight_low = recon_weight_low
        self.cross_recon_weight_low = cross_recon_weight_low
        self.cross_recon_weight_high = cross_recon_weight_high
        self.smooth_weight = smooth_weight
        self.equal_r_weight = equal_r_weight

    @property
    def use_smooth(self) -> bool:
        return self.smooth_weight > 0

    @property
    def self_recon_weight(self) -> float:
        return 0.0

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).to(input_tensor.device)
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = F.conv2d(input_tensor, kernel, stride=1, padding=1).abs()
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def forward(self, R_low, R_high, L_low, L_high, I_low, I_high, L_R_low=None, L_R_high=None):
        zero = torch.tensor(0.0, device=R_low.device)
        L_low_3 = torch.cat((L_low, L_low, L_low), dim=1)
        L_high_3 = torch.cat((L_high, L_high, L_high), dim=1)

        self.recon_loss_low = F.l1_loss(R_low * L_low_3, I_low)
        self.recon_loss_high = F.l1_loss(R_high * L_high_3, I_high)
        loss_recon = self.recon_weight_high * self.recon_loss_high + self.recon_weight_low * self.recon_loss_low

        self.recon_loss_crs_low = F.l1_loss(R_high * L_low_3, I_low)
        self.recon_loss_crs_high = F.l1_loss(R_low * L_high_3, I_high)
        loss_cross = self.cross_recon_weight_low * self.recon_loss_crs_low + self.cross_recon_weight_high * self.recon_loss_crs_high

        if self.equal_r_weight > 0:
            self.equal_R_loss = F.l1_loss(R_low, R_high.detach())
        else:
            self.equal_R_loss = zero

        if self.use_smooth:
            self.Ismooth_loss_low = self.smooth(L_low, R_low)
            self.Ismooth_loss_high = self.smooth(L_high, R_high)
            loss_smooth = self.Ismooth_loss_low + self.Ismooth_loss_high
        else:
            self.Ismooth_loss_low = zero
            self.Ismooth_loss_high = zero
            loss_smooth = zero

        self.loss_Decom = loss_recon + loss_cross + self.smooth_weight * loss_smooth + self.equal_r_weight * self.equal_R_loss

        return (
            self.loss_Decom,
            self.recon_loss_low + self.recon_loss_high,
            zero,
            zero,
            loss_smooth,
            zero,
        )


class UnpairedLoss(nn.Module):
    """当前项目风格的非配对分解损失（兼容现有训练日志字段）。"""

    def __init__(self, recon_weight=1, anchor_weight=0.05, bdsp_weight=0.05,
                 smooth_weight=0, self_recon_weight=0.05):
        super().__init__()
        self.recon_weight = recon_weight
        self.anchor_weight = anchor_weight
        self.bdsp_weight = bdsp_weight
        self.smooth_weight = smooth_weight
        self.self_recon_weight = self_recon_weight

    @property
    def use_smooth(self):
        return self.smooth_weight > 0

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).to(input_tensor.device)
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = F.conv2d(input_tensor, kernel, stride=1, padding=1).abs()
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def forward(self, R_low, R_high, L_low, L_high, I_low, I_high, L_R_low, L_R_high):
        zero = torch.tensor(0.0, device=R_low.device)
        L_low_3 = torch.cat((L_low, L_low, L_low), dim=1)
        L_high_3 = torch.cat((L_high, L_high, L_high), dim=1)

        self.recon_loss_low = F.l1_loss(R_low * L_low_3, I_low)
        self.recon_loss_high = F.l1_loss(R_high * L_high_3, I_high)
        loss_recon = self.recon_loss_low + self.recon_loss_high

        max_low = torch.max(I_low, dim=1, keepdim=True)[0]
        max_high = torch.max(I_high, dim=1, keepdim=True)[0]
        self.recon_loss_anchor_high = F.l1_loss(max_high, L_high)
        self.recon_loss_anchor_low = F.l1_loss(max_low, L_low)
        loss_anchor = self.recon_loss_anchor_high + self.recon_loss_anchor_low

        if self.bdsp_weight > 0:
            self.bdsp_loss = F.l1_loss(BDSP_Face(R_low), BDSP_Face(I_low)) + F.l1_loss(BDSP_Face(R_high), BDSP_Face(I_high))
        else:
            self.bdsp_loss = zero

        if self.self_recon_weight > 0:
            self.self_recon_loss = F.l1_loss(L_R_low, L_R_high)
        else:
            self.self_recon_loss = zero

        if self.use_smooth:
            self.Ismooth_loss_low = self.smooth(L_low, R_low)
            self.Ismooth_loss_high = self.smooth(L_high, R_high)
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
