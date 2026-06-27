"""
train.py

Retinex 分解模型训练入口。

支持两种配置方式：
1. 命令行参数（向后兼容）
2. YAML 配置文件（推荐）

用法示例：
    # 使用配置文件（推荐）
    python train.py --config configs/paired/lol_exp1.yaml

    # 使用命令行参数（向后兼容）
    python train.py --data-path /path/to/dataset --epochs 300 --batch-size 2 --lr 0.0001
"""

import os
import sys
import argparse
import datetime
import time

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from data import MyDataSet, UnpairedDataSet, PureLowDataSet, PureLowSingleDataSet, transforms as T
from models import RetinexPointRaw, RetinexPixelClassic, RetinexPixelTrans
from utils import read_data, read_pure_low_data, train_step, evaluate, create_lr_scheduler, load_config, _build_loss_function

class Tee:
    """将 stdout 同时输出到终端和日志文件。"""

    def __init__(self, log_path, stream):
        self._stream = stream
        self._file = open(log_path, 'a', encoding='utf-8', buffering=1)
        self._line_buf = ''  # 收集当前行内容，用于日志去重

    def write(self, text):
        self._stream.write(text)
        # 处理 \r 进度行：终端保留 \r，日志文件按换行写入
        for ch in text:
            if ch == '\r':
                self._line_buf = ''
            elif ch == '\n':
                if self._line_buf:
                    self._file.write(self._line_buf + '\n')
                self._line_buf = ''
            else:
                self._line_buf += ch

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        # flush 残留内容（如 end='' 的最后一行）
        if self._line_buf:
            self._file.write(self._line_buf + '\n')
            self._line_buf = ''
        self._file.close()

def generate_experiment_name(cfg):
    """
    根据配置生成实验目录名。

    auto_name=false 时：使用 experiment.name（必填）。
    auto_name=true 时：{dataset}_{mode}[_{losses}]
    其中 dataset 取 data.path 末段，mode 取 loss.mode 或 data.mode，
    losses 只包含非零权重，格式为 {值}{缩写}，用 _ 连接。
    """
    # 损失字段 -> 缩写映射（按固定顺序输出）
    LOSS_ABBR = [
        ("recon_weight",              "r"),
        ("recon_weight_high",         "rh"),
        ("recon_weight_low",          "rl"),
        ("cross_recon_weight_high",   "crh"),
        ("cross_recon_weight_low",    "crl"),
        ("equal_r_weight",            "er"),
        ("anchor_weight",             "anchor"),
        ("bdsp_weight",               "bdsp"),
        ("smooth_weight",             "sm"),
        ("self_recon_weight",         "sr"),
        ("reflect_weight",            "ref"),
    ]

    exp_cfg = cfg.get("experiment", {})
    auto_name = exp_cfg.get("auto_name", False)
    if not auto_name:
        name = exp_cfg.get("name", "")
        if not name:
            raise ValueError(
                "experiment.name 未设置。请在配置中指定 experiment.name，"
                "或设置 experiment.auto_name: true 自动生成。"
            )
        return name

    # ---- auto_name=true：从配置自动组装 ----
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})

    # dataset: 取 data.path 最后一段目录名
    data_path = data_cfg.get("path", "unknown")
    dataset = os.path.basename(data_path.rstrip("/\\"))

    # mode: 优先 loss.mode，缺省 data.mode
    mode = loss_cfg.get("mode", "unknown")

    # 非零损失权重
    parts = []
    for key, abbr in LOSS_ABBR:
        val = loss_cfg.get(key, 0)
        if val:
            parts.append(f"{val}{abbr}")

    return "_".join([dataset, mode] + parts)


def main(args):
    # ---- 加载配置 ----
    if args.config:
        cfg = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        # 从命令行参数构建配置（向后兼容）
        cfg = {
            "experiment": {
                "name": "decom_lol",
                "auto_name": False,
            },
            "data": {
                "path": args.data_path,
                "mode": args.mode,
                "crop_size": 256,
                "batch_size": args.batch_size,
                "num_workers": 0
            },
            "model": {
                "name": "RetinexPointRaw",
                "use_dp": args.use_dp,
                "gpu_id": args.gpu_id
            },
            "training": {
                "epochs": args.epochs,
                "lr": args.lr,
                "warmup_iterations": 0,
                "save_ckpt_interval": 5000,
                "log_interval": 1000,
                "eval": {
                    "eval_interval": 500,
                    "save_best_ckpt": True
                }
            },
            "loss": {
                "mode": "unpaired_point",
                "recon_weight": 1,
                "anchor_weight": 0.05,
                "bdsp_weight": 0.05,
                "smooth_weight": 0,
                "self_recon_weight": 0.05
            },
            "resume": {
                "checkpoint": args.resume,
                "weights": args.weights
            },
            "device": args.device
        }
        print("Using command-line arguments (no config file specified)")

    # ---- 提取配置 ----
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    if "mode" not in loss_cfg:
        raise ValueError(
            "loss.mode must be explicitly set in config. Choose from:\n"
            "  paired_point, paired_pixel,\n"
            "  unpaired_point, unpaired_pixel,\n"
            "  pure_low_double_point, pure_low_double_pixel,\n"
            "  pure_low_single_point, pure_low_single_pixel"
        )
    resume_cfg = cfg.get("resume", {})
    device_str = cfg.get("device", "cuda")

    # ---- 设备设置 ----
    if "cuda" in device_str:
        os.environ['CUDA_VISIBLE_DEVICES'] = model_cfg.get("gpu_id", "0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ---- 生成实验目录名 ----
    experiment_name = generate_experiment_name(cfg)
    model_name = model_cfg.get("name", "RetinexPointRaw")
    data_mode = data_cfg.get("mode", "unknown")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = f"./experiments/{model_name}/{data_mode}/{experiment_name}_{timestamp}"
    os.makedirs(filefold_path, exist_ok=True)

    file_img_path = os.path.join(filefold_path, "img")
    os.makedirs(file_img_path, exist_ok=True)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path, exist_ok=True)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path, exist_ok=True)

    # 将 stdout 重定向到终端 + 日志文件
    log_file_path = os.path.join(filefold_path, "train.log")
    sys.stdout = Tee(log_file_path, sys.__stdout__)
    sys.stderr = Tee(log_file_path, sys.__stderr__)

    # 保存配置到实验目录（便于复现）
    import yaml
    config_save_path = os.path.join(filefold_path, "config.yaml")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"Config saved to: {config_save_path}")

    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Directory:  {filefold_path}")
    print(f"{'='*60}\n")

    # ---- TensorBoard ----
    tb_writer = SummaryWriter(log_dir=file_log_path)

    best_valloss = 1e5

    # ---- 加载数据 ----
    data_path = data_cfg.get("path", "")
    data_mode = data_cfg["mode"]
    if data_mode in ("pure_low_single", "pure_low_double"):
        train_low_path, val_low_path = read_pure_low_data(data_path)
        train_high_path, val_high_path = [], []
    else:
        train_low_path, train_high_path, val_low_path, val_high_path = read_data(data_path, mode=data_mode)

    crop_size = data_cfg.get("crop_size", 256)
    if data_mode == "paired":
        data_transform = {
            "train": T.Compose([T.RandomCrop(crop_size),
                                T.RandomHorizontalFlip(0.5),
                                T.RandomVerticalFlip(0.5),
                                T.ToTensor()]),
            "val": T.Compose([T.ToTensor()])
        }
        train_dataset = MyDataSet(images_low_path=train_low_path,
                                  images_high_path=train_high_path,
                                  transform=data_transform["train"])
        val_dataset = MyDataSet(images_low_path=val_low_path,
                                images_high_path=val_high_path,
                                transform=data_transform["val"])
    elif data_mode == "pure_low_double":
        from torchvision import transforms as torchvision_T
        from data.transforms import RandomGamma, RandomBrightness
        data_transform = {
            "train": torchvision_T.Compose([torchvision_T.RandomCrop(crop_size),
                                              torchvision_T.RandomHorizontalFlip(0.5),
                                              torchvision_T.RandomVerticalFlip(0.5),
                                              torchvision_T.ToTensor()]),
            "val": torchvision_T.Compose([torchvision_T.ToTensor()])
        }
        # 光度增强：仅训练时启用，对两个 view 独立应用
        aug_cfg = data_cfg.get("photometric_augment", {})
        photo_transform = None
        if aug_cfg.get("enabled", False):
            from torchvision import transforms as torchvision_T2
            aug_list = []
            if aug_cfg.get("gamma_range"):
                aug_list.append(RandomGamma(tuple(aug_cfg["gamma_range"])))
            if aug_cfg.get("brightness_range"):
                aug_list.append(RandomBrightness(tuple(aug_cfg["brightness_range"])))
            if aug_list:
                photo_transform = torchvision_T2.Compose(aug_list)
        train_dataset = PureLowDataSet(images_low_path=train_low_path,
                                        transform=data_transform["train"],
                                        photometric_transform=photo_transform)
        val_dataset = PureLowDataSet(images_low_path=val_low_path,
                                      transform=data_transform["val"],
                                      photometric_transform=None)
    elif data_mode == "pure_low_single":
        from torchvision import transforms as torchvision_T
        data_transform = {
            "train": torchvision_T.Compose([torchvision_T.RandomCrop(crop_size),
                                              torchvision_T.RandomHorizontalFlip(0.5),
                                              torchvision_T.RandomVerticalFlip(0.5),
                                              torchvision_T.ToTensor()]),
            "val": torchvision_T.Compose([torchvision_T.ToTensor()])
        }
        train_dataset = PureLowSingleDataSet(images_low_path=train_low_path,
                                              transform=data_transform["train"])
        val_dataset = PureLowSingleDataSet(images_low_path=val_low_path,
                                            transform=data_transform["val"])
    else:
        from torchvision import transforms as torchvision_T
        data_transform = {
            "train": torchvision_T.Compose([torchvision_T.RandomCrop(crop_size),
                                              torchvision_T.RandomHorizontalFlip(0.5),
                                              torchvision_T.RandomVerticalFlip(0.5),
                                              torchvision_T.ToTensor()]),
            "val": torchvision_T.Compose([torchvision_T.ToTensor()])
        }
        train_dataset = UnpairedDataSet(images_low_path=train_low_path,
                                        images_high_path=train_high_path,
                                        transform=data_transform["train"])
        val_dataset = UnpairedDataSet(images_low_path=val_low_path,
                                      images_high_path=val_high_path,
                                      transform=data_transform["val"])

    batch_size = data_cfg.get("batch_size", 2)
    num_workers = data_cfg.get("num_workers", 0)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=num_workers,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)

    # ---- 构建模型 ----
    model_name = model_cfg.get("name", "RetinexPointRaw")
    if model_name == "RetinexPixelClassic":
        model = RetinexPixelClassic(
            dim=model_cfg.get("dim", 24),
            l_channel=model_cfg.get("l_channel", 32),
        ).to(device)
    elif model_name == "RetinexPixelTrans":
        model = RetinexPixelTrans(
            dim=model_cfg.get("dim", 24),
            l_heads=model_cfg.get("l_heads", 1),
            l_ffn_expansion=model_cfg.get("l_ffn_expansion", 2.66),
        ).to(device)
    else:
        model = RetinexPointRaw().to(device)
    use_dp = model_cfg.get("use_dp", False)
    if use_dp:
        model = torch.nn.DataParallel(model).cuda()

    # ---- 加载权重 ----
    weights_path = resume_cfg.get("weights", "")
    if weights_path:
        assert os.path.exists(weights_path), f"weights file: '{weights_path}' not exist."
        weights_dict = torch.load(weights_path, map_location=device)["model"]
        print(model.load_state_dict(weights_dict, strict=False))

    # ---- 优化器和学习率调度器 ----
    lr = train_cfg.get("lr", 0.0001)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5E-5)

    # ---- 训练步数 ----
    iterations_per_epoch = len(train_loader)
    max_iterations = train_cfg.get("max_iterations", 0)
    if max_iterations <= 0:
        # 兼容旧配置：从 max_epochs / epochs 推算
        max_epochs = train_cfg.get("max_epochs", train_cfg.get("epochs", 300))
        max_iterations = max_epochs * iterations_per_epoch
        print(f"从 max_epochs={max_epochs} 推算: max_iterations={max_iterations}")
    else:
        print(f"max_iterations: {max_iterations} (iterations_per_epoch={iterations_per_epoch})")

    # ---- 学习率调度器 ----
    warmup_iterations = train_cfg.get("warmup_iterations", 0)
    lr_scheduler = create_lr_scheduler(optimizer, max_iterations, warmup_iterations)

    # ---- 恢复训练 ----
    start_iter = 0
    checkpoint_path = resume_cfg.get("checkpoint", "")
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_iter = checkpoint.get('global_iter', checkpoint.get('epoch', 0) * iterations_per_epoch) + 1
        print(f"Resumed from checkpoint: step {start_iter}")

    # ---- 评估/保存间隔 ----
    eval_cfg = train_cfg.get("eval", {})
    eval_interval = eval_cfg.get("eval_interval", 500)
    save_img_interval = eval_cfg.get("save_img_interval", eval_interval * 4)
    save_best_ckpt = eval_cfg.get("save_best_ckpt", True)
    save_ckpt_interval = train_cfg.get("save_ckpt_interval", 5000)
    log_interval = train_cfg.get("log_interval", 1000)

    print(f"eval_interval: {eval_interval} steps")
    print(f"log_interval: {log_interval} steps")
    print(f"save_ckpt_interval: {save_ckpt_interval} steps")
    print(f"save_img_interval: {save_img_interval} steps")

    # ---- 校验 interval 约束 ----
    if eval_interval <= 0:
        raise ValueError(f"eval_interval must be > 0, got {eval_interval}")
    if save_img_interval < 0:
        raise ValueError(f"save_img_interval must be >= 0, got {save_img_interval}")
    if save_img_interval > 0 and save_img_interval % eval_interval != 0:
        raise ValueError(
            f"save_img_interval ({save_img_interval}) must be a multiple of eval_interval ({eval_interval})"
        )

    # ---- 构建损失函数（复用，不重复构建）----
    loss_function = _build_loss_function(loss_cfg)
    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    # ---- 训练循环（step-based）----
    global_iter = start_iter
    epoch = 0
    data_iter = iter(train_loader)
    accum_loss = torch.zeros(6, device=device)
    accum_count = 0
    train_start = time.perf_counter()

    while global_iter < max_iterations:
        # 获取下一个 batch，epoch 结束时重新 shuffle
        try:
            data = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(train_loader)
            data = next(data_iter)

        # 单步训练
        loss_vals = train_step(model, optimizer, loss_function, data, device, lr_scheduler)
        accum_loss += torch.tensor(loss_vals, device=device)
        accum_count += 1
        global_iter += 1
        lr = optimizer.param_groups[0]["lr"]

        # 定期打印训练进度
        if global_iter % log_interval == 0:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            avg = accum_loss / accum_count
            loss_names = ["total", "recon", "anchor", "bdsp", "smooth", "self-recon"]
            parts = [f"{loss_names[0]}: {avg[0]:.4f}"]
            for i in range(1, 6):
                if avg[i] > 0:
                    parts.append(f"{loss_names[i]}: {avg[i]:.4f}")
            parts.append(f"lr: {lr:.6f}")
            # 速度和 ETA
            elapsed = time.perf_counter() - train_start
            steps_done = global_iter - start_iter
            if steps_done > 0:
                avg_time = elapsed / steps_done
                remaining = avg_time * (max_iterations - global_iter)
                if remaining >= 3600:
                    eta = f"{remaining/3600:.1f}h"
                elif remaining >= 60:
                    eta = f"{remaining/60:.0f}m"
                else:
                    eta = f"{remaining:.0f}s"
                parts.append(f"time: {avg_time:.3f}s  ETA: {eta}")
            # GPU 显存
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                parts.append(f"mem: {mem_allocated:.2f}GB")
            print(f"[{now}] [iter: {global_iter:>6d}/{max_iterations}] " + " | ".join(parts))

        # ---- 评估 ----
        if global_iter % eval_interval == 0:
            print()  # 换行，避免 tqdm/eval 输出混乱
            save_img = save_img_interval > 0 and global_iter % save_img_interval == 0
            val_loss, val_recon, val_anchor, val_bdsp, val_smooth, val_self_recon, val_psnr = evaluate(
                model=model, data_loader=val_loader, device=device,
                lr=lr, filefold_path=file_img_path,
                loss_function=loss_function, save_images=save_img, global_iter=global_iter)

            # TensorBoard
            train_avg = (accum_loss / accum_count).cpu().numpy()
            tb_writer.add_scalar("train/total_loss", train_avg[0], global_iter)
            tb_writer.add_scalar("train/recon_loss", train_avg[1], global_iter)
            tb_writer.add_scalar("train/anchor_loss", train_avg[2], global_iter)
            tb_writer.add_scalar("train/bdsp_loss", train_avg[3], global_iter)
            tb_writer.add_scalar("train/smooth_loss", train_avg[4], global_iter)
            tb_writer.add_scalar("train/self_recon_loss", train_avg[5], global_iter)
            tb_writer.add_scalar("val/total_loss", val_loss, global_iter)
            tb_writer.add_scalar("val/recon_loss", val_recon, global_iter)
            tb_writer.add_scalar("val/anchor_loss", val_anchor, global_iter)
            tb_writer.add_scalar("val/bdsp_loss", val_bdsp, global_iter)
            tb_writer.add_scalar("val/smooth_loss", val_smooth, global_iter)
            tb_writer.add_scalar("val/self_recon_loss", val_self_recon, global_iter)
            if val_psnr is not None:
                tb_writer.add_scalar("val/psnr", val_psnr, global_iter)

            # 打印评估结果（含 PSNR）
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            psnr_str = f" | PSNR: {val_psnr:.2f}dB" if val_psnr is not None else ""
            print(f"[{now}] [eval  step {global_iter:>6d}] val_loss: {val_loss:.4f}{psnr_str}")

            # 重置训练损失累积
            accum_loss.zero_()
            accum_count = 0

            # 保存最佳模型
            if save_best_ckpt and val_loss < best_valloss:
                best_valloss = val_loss
                best_save_path = os.path.join(file_weights_path, "best_model.pth")
                torch.save({
                    "model": model.module.state_dict() if use_dp else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "global_iter": global_iter,
                    "val_loss": val_loss,
                    "config": cfg
                }, best_save_path)
                best_psnr_str = f" | PSNR: {val_psnr:.2f}dB" if val_psnr is not None else ""
                print(f"Saved best model at step {global_iter} | val_loss: {val_loss:.4f}{best_psnr_str}")

        # ---- 定期保存 checkpoint ----
        if global_iter % save_ckpt_interval == 0:
            save_file = {
                "model": model.module.state_dict() if use_dp else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "global_iter": global_iter,
                "config": cfg
            }
            torch.save(save_file, os.path.join(file_weights_path, f"checkpoint_{global_iter}.pth"))

    print()  # 换行
    tb_writer.close()
    print(f"\nTraining completed. Best validation loss: {best_valloss:.4f}")
    print(f"Results saved to: {filefold_path}")
    # 关闭日志文件
    tee_out = sys.stdout
    tee_err = sys.stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    tee_out.close()
    tee_err.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinex Decomposition Training')

    # 配置文件参数（推荐）
    parser.add_argument('--config', type=str, default='',
                        help='path to YAML config file (recommended)')

    # 命令行参数（向后兼容）
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data-path', type=str,
                        default="datasets/LOLv2")
    parser.add_argument("--mode", type=str, default="paired", choices=["paired", "unpaired"], help="data loading mode")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', default=False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')

    opt = parser.parse_args()
    main(opt)
