"""
data/transforms.py

配对图像数据增强。

所有变换均接收 (image, target) 双张量，保证 low-high 图像同步变换。
支持随机裁剪、水平/垂直翻转、缩放、居中裁剪及转 Tensor。
"""

import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target


class RandomGamma(object):
    """随机 gamma 校正，模拟不同曝光水平。"""
    def __init__(self, gamma_range=(0.7, 1.5)):
        self.gamma_range = gamma_range
    def __call__(self, img):
        gamma = random.uniform(*self.gamma_range)
        return F.adjust_gamma(img, gamma)


class RandomBrightness(object):
    """随机亮度缩放，模拟不同光照强度。"""
    def __init__(self, factor_range=(0.6, 1.4)):
        self.factor_range = factor_range
    def __call__(self, img):
        factor = random.uniform(*self.factor_range)
        return F.adjust_brightness(img, factor)


class RandomGaussianNoise(object):
    """在 tensor 上添加随机高斯噪声。必须在 ToTensor 之后使用。"""
    def __init__(self, std_range=(0.01, 0.05)):
        self.std_range = std_range
    def __call__(self, tensor):
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(tensor) * std
        return (tensor + noise).clamp(0, 1)
