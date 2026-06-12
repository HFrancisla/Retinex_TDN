"""
metrics.py

图像质量评估指标计算。

对两组或多组图像文件夹逐对计算 PSNR、SSIM、LPIPS，
将逐图与平均结果写入 metrics_results.txt。

用法：
    单组：  python metrics.py --pair /path/to/pred /path/to/gt
    多组：  python metrics.py \
                --pair /pred1 /gt1 \
                --pair /pred2 /gt2 \
                --output metrics_results.txt
"""

import argparse
import os

import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 初始化 LPIPS（全局单例，避免重复加载）
_lpips_fn = lpips.LPIPS(net='alex')
_lpips_fn.eval()


def calculate_metrics(img1, img2):
    """计算两张 RGB PIL Image 之间的 PSNR / SSIM / LPIPS。"""
    img1 = np.array(img1).astype(np.float32) / 255.0
    img2 = np.array(img2).astype(np.float32) / 255.0

    psnr_value = psnr(img1, img2, data_range=1.0)
    ssim_value = ssim(img1, img2, channel_axis=-1, data_range=1.0, win_size=7)

    img1_tensor = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    img2_tensor = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) * 2 - 1

    with torch.no_grad():
        lpips_value = _lpips_fn(img1_tensor, img2_tensor).item()

    return psnr_value, ssim_value, lpips_value


def process_folders(folder_a, folder_b, output_file, verbose=True):
    """
    对两个文件夹中的同名图像逐对计算指标。

    结果写入 folder_a/output_file。
    verbose=True 时打印逐图详情与调试信息。
    """
    exts = {'.png', '.jpg', '.jpeg'}
    files_a = {f for f in os.listdir(folder_a) if os.path.splitext(f)[-1].lower() in exts}
    files_b = {f for f in os.listdir(folder_b) if os.path.splitext(f)[-1].lower() in exts}
    common_files = sorted(files_a & files_b)

    if verbose:
        print(f"folder_a: {folder_a}  ({len(files_a)} files)")
        print(f"folder_b: {folder_b}  ({len(files_b)} files)")
        print(f"common files: {len(common_files)}")
        if common_files:
            print(f"sample: {common_files[:5]}")

    total_psnr = total_ssim = total_lpips = 0.0
    results = []

    for file in common_files:
        path_a = os.path.join(folder_a, file)
        path_b = os.path.join(folder_b, file)

        try:
            img_a = Image.open(path_a).convert('RGB')
            img_b = Image.open(path_b).convert('RGB')

            if img_a.size != img_b.size:
                print(f"  skip {file}: size mismatch {img_a.size} vs {img_b.size}")
                continue

            p, s, l = calculate_metrics(img_a, img_b)
            results.append((file, p, s, l))
            total_psnr += p
            total_ssim += s
            total_lpips += l

            if verbose:
                print(f"  {file}: PSNR={p:.4f}  SSIM={s:.4f}  LPIPS={l:.4f}")

        except Exception as e:
            print(f"  error processing {file}: {e}")

    count = len(results)
    avg_psnr = total_psnr / count if count else 0
    avg_ssim = total_ssim / count if count else 0
    avg_lpips = total_lpips / count if count else 0

    output_path = os.path.join(folder_a, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("File\tPSNR\tSSIM\tLPIPS\n")
        for file, p, s, l in results:
            f.write(f"{file}\t{p:.4f}\t{s:.4f}\t{l:.4f}\n")
        f.write(f"\nAverage\t{avg_psnr:.4f}\t{avg_ssim:.4f}\t{avg_lpips:.4f}\n")

    print(f"results saved to: {output_path}")
    print(f"Average ({count} images): PSNR={avg_psnr:.4f}  SSIM={avg_ssim:.4f}  LPIPS={avg_lpips:.4f}")

    return avg_psnr, avg_ssim, avg_lpips


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图像质量评估：PSNR / SSIM / LPIPS")
    parser.add_argument('--pair', nargs=2, action='append', metavar=('PRED', 'GT'),
                        required=True, help='预测图与真值图文件夹，可多次指定')
    parser.add_argument('--output', default='metrics_results.txt',
                        help='结果文件名（写入各 PRED 文件夹内）')
    args = parser.parse_args()

    for i, (folder_a, folder_b) in enumerate(args.pair):
        if len(args.pair) > 1:
            print(f"\n{'='*60}\n  Pair {i+1}/{len(args.pair)}\n{'='*60}")
        process_folders(folder_a, folder_b, args.output)
