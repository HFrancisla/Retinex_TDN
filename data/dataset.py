"""
data/dataset.py

配对与非配对图像数据集。

MyDataSet 加载成对图像，UnpairedDataSet 独立随机采样低光与正常光图像。
"""

import random

from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, images_low_path: list, images_high_path: list, transform=None):
        self.images_low_path = images_low_path
        self.images_high_path = images_high_path
        self.transform = transform

    def __len__(self):
        return len(self.images_low_path)

    def __getitem__(self, item):
        img = Image.open(self.images_low_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_low_path[item]))
        img_ref = Image.open(self.images_high_path[item])
        if img_ref.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_high_path[item]))

        if self.transform is not None:
            img, img_ref = self.transform(img, img_ref)

        return img, img_ref

    @staticmethod
    def collate_fn(batch):
        images, images_ref = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        images_ref = torch.stack(images_ref, dim=0)
        return images, images_ref


class UnpairedDataSet(Dataset):
    """
    非配对图像数据集。

    低光与正常光图像各自独立加载，__getitem__ 为每张图随机采样配对对象。
    增强变换对两张图分别独立施加（无同步裁剪/翻转）。
    """

    def __init__(self, images_low_path: list, images_high_path: list, transform=None):
        self.images_low_path = images_low_path
        self.images_high_path = images_high_path
        self.transform = transform

    def __len__(self):
        return len(self.images_low_path)

    def __getitem__(self, item):
        img = Image.open(self.images_low_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_low_path[item]))
        high_idx = random.randint(0, len(self.images_high_path) - 1)
        img_ref = Image.open(self.images_high_path[high_idx])
        if img_ref.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_high_path[high_idx]))

        if self.transform is not None:
            img = self.transform(img)
            img_ref = self.transform(img_ref)

        return img, img_ref

    @staticmethod
    def collate_fn(batch):
        images, images_ref = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        images_ref = torch.stack(images_ref, dim=0)
        return images, images_ref


class PureLowDataSet(Dataset):
    """
    Pure-low 模式数据集。

    仅加载 low 图像，并为每张图构造两个独立增强视图，
    用于无 normal-light 监督的自监督分解训练。
    """

    def __init__(self, images_low_path: list, transform=None, photometric_transform=None):
        self.images_low_path = images_low_path
        self.transform = transform
        self.photometric_transform = photometric_transform

    def __len__(self):
        return len(self.images_low_path)

    def __getitem__(self, item):
        img = Image.open(self.images_low_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_low_path[item]))

        # 光度增强：对两个 view 独立应用（在空间增强之前，PIL 域）
        if self.photometric_transform is not None:
            img1 = self.photometric_transform(img)
            img2 = self.photometric_transform(img)
        else:
            img1 = img
            img2 = img

        # 空间增强：独立随机裁剪/翻转
        if self.transform is not None:
            view1 = self.transform(img1)
            view2 = self.transform(img2)
        else:
            view1 = img1
            view2 = img2

        return view1, view2

    @staticmethod
    def collate_fn(batch):
        views1, views2 = tuple(zip(*batch))
        views1 = torch.stack(views1, dim=0)
        views2 = torch.stack(views2, dim=0)
        return views1, views2


class PureLowSingleDataSet(Dataset):
    """
    Pure-low 单视图数据集。

    仅加载 low 图像，对每张图执行一次增强，返回单张图，
    由训练/验证循环单独处理，避免 single 模式下的重复 loss 计算。
    """

    def __init__(self, images_low_path: list, transform=None):
        self.images_low_path = images_low_path
        self.transform = transform

    def __len__(self):
        return len(self.images_low_path)

    def __getitem__(self, item):
        img = Image.open(self.images_low_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_low_path[item]))

        if self.transform is not None:
            view = self.transform(img)
        else:
            view = img

        return view

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch, dim=0)
