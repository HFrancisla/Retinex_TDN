import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
import torch
import lpips

# 初始化 LPIPS 模型
lpips_fn = lpips.LPIPS(net='alex')
lpips_fn.eval()

def calculate_metrics(img1, img2):
    img1 = np.array(img1).astype(np.float32) / 255.0
    img2 = np.array(img2).astype(np.float32) / 255.0

    print(f"img1 shape: {img1.shape}, min: {img1.min():.4f}, max: {img1.max():.4f}")
    print(f"img2 shape: {img2.shape}, min: {img2.min():.4f}, max: {img2.max():.4f}")

    psnr_value = psnr(img1, img2, data_range=1.0)
    ssim_value = ssim(img1, img2, channel_axis=-1, data_range=1.0, win_size=7)

    img1_tensor = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    img2_tensor = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) * 2 - 1

    with torch.no_grad():
        lpips_value = lpips_fn(img1_tensor, img2_tensor).item()

    return psnr_value, ssim_value, lpips_value

def process_folders(folder_a, folder_b, output_file):
    files_a = {f for f in os.listdir(folder_a) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    files_b = {f for f in os.listdir(folder_b) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    common_files = sorted(files_a.intersection(files_b))

    print("folder_a:", folder_a)
    print("folder_b:", folder_b)
    print("files in folder_a:", len(files_a))
    print("files in folder_b:", len(files_b))
    print("common files:", len(common_files))
    print("sample common files:", common_files[:10])

    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    results = []

    for file in common_files:
        path_a = os.path.join(folder_a, file)
        path_b = os.path.join(folder_b, file)

        try:
            img_a = Image.open(path_a).convert('RGB')
            img_b = Image.open(path_b).convert('RGB')

            if img_a.size != img_b.size:
                print(f"skip {file}: size mismatch {img_a.size} vs {img_b.size}")
                continue

            psnr_value, ssim_value, lpips_value = calculate_metrics(img_a, img_b)
            results.append((file, psnr_value, ssim_value, lpips_value))
            total_psnr += psnr_value
            total_ssim += ssim_value
            total_lpips += lpips_value

            print(f"{file}: PSNR={psnr_value:.4f}, SSIM={ssim_value:.4f}, LPIPS={lpips_value:.4f}")

        except Exception as e:
            print(f"error processing {file}: {e}")

    count = len(results)
    avg_psnr = total_psnr / count if count > 0 else 0
    avg_ssim = total_ssim / count if count > 0 else 0
    avg_lpips = total_lpips / count if count > 0 else 0

    output_path = os.path.join(folder_a, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("File\tPSNR\tSSIM\tLPIPS\n")
        for file, psnr_value, ssim_value, lpips_value in results:
            f.write(f"{file}\t{psnr_value:.4f}\t{ssim_value:.4f}\t{lpips_value:.4f}\n")
        f.write(f"\nAverage\t{avg_psnr:.4f}\t{avg_ssim:.4f}\t{avg_lpips:.4f}\n")

    print("results saved to:", output_path)
    print(f"Average: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")

if __name__ == "__main__":
    folder_a = r"E:\XUD\Diff-Retinex-main\model\Diff_TDN-Unet-dot-unpair-NEW\results\LOL_high_eval\LLxHR"
    folder_b = r"E:\XUD\datasets\NTM\test\low_GRI"
    output_file = "metrics_results.txt"
    process_folders(folder_a, folder_b, output_file)