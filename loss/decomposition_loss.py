"""
loss/decomposition_loss.py

Retinex 分解损失函数。

包含重建损失（R * L ≈ I）、交叉光照一致性损失、BDSP 结构保持损失
和光照平滑损失，各分量加权求和返回总损失。
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


class Decom_Loss(nn.Module):
    def __init__(self, recon_weight=20, cross_recon_weight=1, bdsp_weight=1,
                 smooth_weight=0, self_recon_weight=1):
        super().__init__()
        self.recon_weight = recon_weight
        self.cross_recon_weight = cross_recon_weight
        self.bdsp_weight = bdsp_weight
        self.smooth_weight = smooth_weight
        self.self_recon_weight = self_recon_weight

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
        L_low_3 = torch.cat((L_low, L_low, L_low), dim=1)
        L_high_3 = torch.cat((L_high, L_high, L_high), dim=1)

        # Reconstruction loss: R * L ≈ I
        self.recon_loss_low = F.l1_loss(R_low * L_low_3, I_low)
        self.recon_loss_high = F.l1_loss(R_high * L_high_3, I_high)

        # Cross-illumination consistency loss
        max_low = torch.max(I_low, dim=1, keepdim=True)[0]
        max_high = torch.max(I_high, dim=1, keepdim=True)[0]
        self.recon_loss_crs_low = F.l1_loss(max_high, L_high)
        self.recon_loss_crs_high = F.l1_loss(max_low, L_low)

        # BDSP structure preservation loss
        self.bdsp_loss = F.l1_loss(BDSP_Face(R_low), BDSP_Face(I_low)) + F.l1_loss(BDSP_Face(R_high), BDSP_Face(I_high))

        # Self-reconstruction constraint
        self.self_recon_loss = F.l1_loss(L_R_low, L_R_high)

        # Illumination smoothness loss
        self.Ismooth_loss_low = self.smooth(L_low, R_low)
        self.Ismooth_loss_high = self.smooth(L_high, R_high)

        # Weighted total loss
        self.loss_Decom = (self.recon_weight * (self.recon_loss_low + self.recon_loss_high) +
                           self.cross_recon_weight * (self.recon_loss_crs_low + self.recon_loss_crs_high) +
                           self.bdsp_weight * self.bdsp_loss +
                           self.self_recon_weight * self.self_recon_loss +
                           self.smooth_weight * (self.Ismooth_loss_low + self.Ismooth_loss_high))

        return (self.loss_Decom,
                self.recon_loss_low + self.recon_loss_high,
                self.recon_loss_crs_low + self.recon_loss_crs_high,
                self.bdsp_loss,
                self.Ismooth_loss_low + self.Ismooth_loss_high,
                self.self_recon_loss)
