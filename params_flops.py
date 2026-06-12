import torch
from PIL import Image
from torchvision import transforms
from TDN_network import DecomNet as create_model
import numpy as np
import cv2
import os

# ==================== 自动计算 Params & FLOPs ====================
from thop import profile


def print_model_params_flops(model, input_size=(3, 400, 600), device="cuda"):
    """计算并打印模型参数量和 FLOPs"""
    dummy_input = torch.randn(1, *input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    print("-" * 50)
    print(f"Model Params: {params / 1e6:.2f} M")
    print(f"Model FLOPs: {flops / 1e9:.2f} G")
    print("-" * 50)


# =================================================================

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.ToTensor()

    # 路径配置
    root_low = r"D:\Datasets\NTM609\test\low"
    root_high = r"D:\Datasets\NTM609\test\high"
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

    # ==================== 模型加载 + 测试 Params & FLOPs ====================
    model = create_model().to(device)
    #model_weight_path = r"D:\2024_XuD\lowlight\Diff-Retinex-main\model\Diff_TDN-Unet-dot-unpair-NEW3_pool-DWT10_FSA\experiments\TDN_train_20260419-125653\weights\checkpoint_Diff_TDN.pth"

    # 修复你之前的 temperature_freq 维度错误
    # weights_dict = torch.load(model_weight_path, map_location=device)['model']
    #model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])

    model.eval()
    # 测试模型参数量与计算量
    print_model_params_flops(model, input_size=(3, 256, 256), device=device)

    # ======================================================================

    # 统一预处理函数
    def process_image(img_path):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img = img.resize((w - w % 8, h - h % 8))  # 缩放到8的倍数
        return data_transform(img).unsqueeze(0).to(device)

    # 保存图片函数（已修复归一化bug）
    def savepic(tensor, name, flag):
        arr = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        arr = arr[:, :, ::-1]  # RGB -> BGR for cv2

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