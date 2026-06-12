import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
import torch
import lpips

# 初始化 LPIPS 模型
lpips_fn = lpips.LPIPS(net='alex')  # 使用 AlexNet 特征

def calculate_metrics(img1, img2):
    # 转换为 NumPy 数组
    img1 = np.array(img1).astype(np.float32) / 255.0
    img2 = np.array(img2).astype(np.float32) / 255.0

    # 检查图像形状和内容
    print(f"img1 shape: {img1.shape}, min: {img1.min()}, max: {img1.max()}")
    print(f"img2 shape: {img2.shape}, min: {img2.min()}, max: {img2.max()}")

    # PSNR
    psnr_value = psnr(img1, img2, data_range=1.0)

    # SSIM (确保正确设置 win_size 和 channel_axis)
    ssim_value = ssim(img1, img2, channel_axis=-1, data_range=1.0, win_size=7)

    # LPIPS 需要将图像转换为张量并归一化到 [-1, 1]
    img1_tensor = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    img2_tensor = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    lpips_value = lpips_fn(img1_tensor, img2_tensor).item()

    return psnr_value, ssim_value, lpips_value

def process_folders(folder_a, folder_b, output_file):
    # 获取文件列表
    files_a = {f for f in os.listdir(folder_a) if f.endswith(('PNG', 'png', 'jpg', 'jpeg'))}
    files_b = {f for f in os.listdir(folder_b) if f.endswith(('PNG', 'png', 'jpg', 'jpeg'))}
    common_files = files_a.intersection(files_b)

    # 初始化指标
    total_psnr, total_ssim, total_lpips = 0, 0, 0
    results = []

    # 遍历文件
    for file in common_files:
        img_a = Image.open(os.path.join(folder_a, file)).convert('RGB')
        img_b = Image.open(os.path.join(folder_b, file)).convert('RGB')

        # 计算指标
        psnr_value, ssim_value, lpips_value = calculate_metrics(img_a, img_b)
        results.append((file, psnr_value, ssim_value, lpips_value))
        total_psnr += psnr_value
        total_ssim += ssim_value
        total_lpips += lpips_value

    # 计算平均值
    count = len(results)
    avg_psnr = total_psnr / count if count > 0 else 0
    avg_ssim = total_ssim / count if count > 0 else 0
    avg_lpips = total_lpips / count if count > 0 else 0

    # 写入结果文件
    with open(os.path.join(folder_a, output_file), 'w') as f:
        f.write("File\tPSNR\tSSIM\tLPIPS\n")
        for file, psnr_value, ssim_value, lpips_value in results:
            f.write(f"{file}\t{psnr_value:.4f}\t{ssim_value:.4f}\t{lpips_value:.4f}\n")
        f.write(f"\nAverage\t{avg_psnr:.4f}\t{avg_ssim:.4f}\t{avg_lpips:.4f}\n")

if __name__ == "__main__":
    folder_LLxLR = r"E:\XUD\Diff-Retinex-main\model\retinexLR-test3max\results\LOL_v2_eval\LLxLR"
    folder_NTM = r"E:\XUD\datasets\LOLv2-real\Test\low"  # 替换为实际路径

    folder_HLxHR = r"E:\XUD\Diff-Retinex-main\model\retinexLR-test3max\results\LOL_v2_eval\HLxHR"
    folder_DTM = r"E:\XUD\datasets\LOLv2-real\Test\high"  # 替换为实际路径

    folder_HLxLR = r"E:\XUD\Diff-Retinex-main\model\retinexLR-test3max\results\LOL_v2_eval\HLxLR"
    folder_high_GRI = r"E:\XUD\datasets\LOLv2-real\Test\high"  # 替换为实际路径

    folder_LLxHR = r"E:\XUD\Diff-Retinex-main\model\retinexLR-test3max\results\LOL_v2_eval\LLxHR"
    folder_low_GRI = r"E:\XUD\datasets\LOLv2-real\Test\low"  # 替换为实际路径
    output_file = "metrics_results.txt"

    process_folders(folder_LLxLR, folder_NTM, output_file)
    process_folders(folder_HLxHR, folder_DTM, output_file)
    process_folders(folder_HLxLR, folder_high_GRI, output_file)
    process_folders(folder_LLxHR, folder_low_GRI, output_file)