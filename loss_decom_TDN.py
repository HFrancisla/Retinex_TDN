import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from BDSP_Face import *

Sobel = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])
Robert = np.array([[0, 0],
                   [-1, 1]])
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)

def gradient(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


def gradient_no_abs(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


class Decom_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def forward(self, R_low, R_high, L_low, L_high, I_low, I_high,L_R_low,L_R_high):
        L_low_3  = torch.cat((L_low, L_low, L_low), dim=1)
        L_high_3 = torch.cat((L_high, L_high, L_high), dim=1)

        # high_GRI = normalize_grad(torch.log(I_low*torch.tensor(255.0)+1))
        # low_GRI = normalize_grad(torch.exp(I_high * (torch.log(torch.tensor(255.0)))))

        self.recon_loss_low  = F.l1_loss(R_low * L_low_3,  I_low)
        self.recon_loss_high = F.l1_loss(R_high * L_high_3, I_high)
        avg_low = torch.max(I_low,dim=1, keepdim=True)[0]#.mean()
        #avg_low = avg_low.expand_as(L_low)
        avg_high = torch.max(I_high, dim=1, keepdim=True)[0]#.mean()
        #avg_high = avg_high.expand_as(L_high)
        self.recon_loss_crs_low  = F.l1_loss(avg_high,L_high) #F.l1_loss(R_high * L_low_3, low_GRI)
        self.recon_loss_crs_high = F.l1_loss(avg_low,L_low) #F.l1_loss(R_low * L_high_3, high_GRI)
        self.equal_R_loss = F.l1_loss(BDSP_Face(R_low), BDSP_Face(I_low))+F.l1_loss(BDSP_Face(R_high), BDSP_Face(I_high)) # torch.tensor(0.0000001) #
        self.r_l_loss = F.l1_loss(L_R_low,L_R_high)

        self.loss_Decom = 20*self.recon_loss_high + 20*self.recon_loss_low + 1 * self.recon_loss_crs_low + \
                          1 * self.recon_loss_crs_high + 1 * self.equal_R_loss + self.r_l_loss

        return self.loss_Decom, 20*(self.recon_loss_low + self.recon_loss_high), (self.recon_loss_crs_low+self.recon_loss_crs_high), self.r_l_loss #0.2*(self.r_l_loss) #self.equal_R_loss

def normalize_grad(gradient_orig):
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm