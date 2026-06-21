"""
analyze_model.py

模型参数量与计算量分析。

使用 thop 库统计 RetinexPointRaw 的参数量（Params）和浮点运算量（FLOPs），
可选加载权重后对指定尺寸输入进行 profile。
"""

import torch
from PIL import Image
from torchvision import transforms
from models import RetinexPointRaw, RetinexPixelClassic, RetinexPixelTrans
import argparse
import numpy as np
import cv2
import os

from thop import profile


def print_model_params_flops(model, input_size=(3, 400, 600), device="cuda"):
    """计算并打印模型参数量和 FLOPs"""
    dummy_input = torch.randn(1, *input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    print("-" * 50)
    print(f"Model Params: {params / 1e6:.2f} M")
    print(f"Model FLOPs: {flops / 1e9:.2f} G")
    print("-" * 50)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.ToTensor()

    # 路径配置
    root_low = "datasets/LOLv2/Test/low"
    root_high = "datasets/LOLv2/Test/high"
    assert os.path.exists(root_low), f"low路径不存在: {root_low}"
    assert os.path.exists(root_high), f"high路径不存在: {root_high}"

    # 加载文件列表
    def get_image_paths(folder):
        exts = [".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"]
        return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[-1] in exts]

    images_low = get_image_paths(root_low)
    images_high = get_image_paths(root_high)
    print(f"低光图数量: {len(images_low)}")
    print(f"高光图数量: {len(images_high)}")

    # 模型加载 + 测试 Params & FLOPs
    model_cls = {"RetinexPointRaw": RetinexPointRaw, "RetinexPixelClassic": RetinexPixelClassic, "RetinexPixelTrans": RetinexPixelTrans}
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="RetinexPointRaw", choices=model_cls.keys())
    args = parser.parse_args()
    model = model_cls[args.model]().to(device)

    model.eval()
    print_model_params_flops(model, input_size=(3, 256, 256), device=device)

    # 统一预处理函数
    def process_image(img_path):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img = img.resize((w - w % 8, h - h % 8))
        return data_transform(img).unsqueeze(0).to(device)

    # 保存图片函数
    def savepic(tensor, name, flag):
        arr = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        arr = arr[:, :, ::-1]

        save_dir = f"./results/LOL_high_eval/{flag}"
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f"{name}.png"), arr)
        print(f"已保存: {flag}/{name}.png")

    # 获取文件名
    def get_name(path):
        return os.path.splitext(os.path.basename(path))[0]


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
