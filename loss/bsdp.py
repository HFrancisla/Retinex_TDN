"""
loss/bsdp.py

BDSP 边缘检测算子。

基于 2x2 block 的对角差分实现边缘响应图，用于 Retinex 分解损失中
约束反射分量 R 的结构一致性。输入为 [0,1] 范围的张量，输出归一化边缘图。
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


def unnormalize(tensor, mean, std, inplace=False):
    if not inplace:
        tensor = tensor.clone()
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def BDSP_Face(img_1):
    """对输入张量计算 BDSP 边缘响应，返回归一化边缘图。"""
    img_1 = unnormalize(img_1, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) * 255
    img = torch.log(img_1 + 1)
    (a, b) = (img.shape[2], img.shape[3])
    in_one_dim = 1
    beta = 0.8

    img1 = nn.functional.pad(img,
                             (in_one_dim, in_one_dim, in_one_dim, in_one_dim),
                             "replicate")

    temp1 = img1[:, 1, 0:a, 0:b]
    temp2 = img1[:, 1, 0:a, 1:b+1]
    temp3 = img1[:, 1, 1:a+1, 0:b]
    temp4 = img1[:, 1, 1:a+1, 1:b+1]

    delete1 = (temp1 - temp4)
    delete2 = (temp2 - temp3)

    result = 2 * (beta - 0.5) * (abs(delete1) + abs(delete2)) + (delete1 + delete2)
    result = torch.atan(4 * result)
    result = result / max(abs(float(result.max())), abs(float(result.min())))
    return Variable(result)
