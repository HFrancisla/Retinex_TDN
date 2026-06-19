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
import argparse
import datetime

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from data import MyDataSet, UnpairedDataSet, PureLowDataSet, transforms as T
from models import DecomNet
from utils import read_data, read_pure_low_data, train_one_epoch, evaluate, create_lr_scheduler, load_config


def generate_experiment_name(cfg):
    """
    根据配置生成实验目录名。

    auto_name=true 时：{name}_{mode}[_{tag}]
    mode 取 loss.mode，缺省跟随 data.mode。
    """
    exp_cfg = cfg.get("experiment", {})
    name = exp_cfg.get("name", "exp")
    auto_name = exp_cfg.get("auto_name", False)
    tag = exp_cfg.get("tag", "")

    if auto_name:
        loss_cfg = cfg.get("loss", {})
        data_cfg = cfg.get("data", {})
        mode = loss_cfg.get("mode", data_cfg.get("mode", "unpaired"))
        name = f"{name}_{mode}"

    if tag:
        name = f"{name}_{tag}"

    return name


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
                "tag": ""
            },
            "data": {
                "path": args.data_path,
                "mode": args.mode,
                "crop_size": 256,
                "batch_size": args.batch_size,
                "num_workers": 0
            },
            "model": {
                "name": "DecomNet",
                "use_dp": args.use_dp,
                "gpu_id": args.gpu_id
            },
            "training": {
                "epochs": args.epochs,
                "lr": args.lr,
                "warmup": True,
                "save_interval": 20,
                "save_best": True
            },
            "loss": {
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
        loss_cfg["mode"] = data_cfg.get("mode", "unpaired")
    resume_cfg = cfg.get("resume", {})
    device_str = cfg.get("device", "cuda")

    # ---- 设备设置 ----
    if "cuda" in device_str:
        os.environ['CUDA_VISIBLE_DEVICES'] = model_cfg.get("gpu_id", "0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ---- 生成实验目录名 ----
    experiment_name = generate_experiment_name(cfg)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = f"./experiments/{experiment_name}_{timestamp}"
    os.makedirs(filefold_path, exist_ok=True)

    file_img_path = os.path.join(filefold_path, "img")
    os.makedirs(file_img_path, exist_ok=True)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path, exist_ok=True)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path, exist_ok=True)

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
    start_epoch = 0

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
        data_transform = {
            "train": torchvision_T.Compose([torchvision_T.RandomCrop(crop_size),
                                              torchvision_T.RandomHorizontalFlip(0.5),
                                              torchvision_T.RandomVerticalFlip(0.5),
                                              torchvision_T.ToTensor()]),
            "val": torchvision_T.Compose([torchvision_T.ToTensor()])
        }
        train_dataset = PureLowDataSet(images_low_path=train_low_path,
                                        transform=data_transform["train"])
        val_dataset = PureLowDataSet(images_low_path=val_low_path,
                                      transform=data_transform["val"])
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
    model = DecomNet().to(device)
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

    epochs = train_cfg.get("epochs", 300)
    warmup = train_cfg.get("warmup", True)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs, warmup=warmup)

    # ---- 恢复训练 ----
    checkpoint_path = resume_cfg.get("checkpoint", "")
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: epoch {start_epoch}")

    # ---- 训练循环 ----
    save_interval = train_cfg.get("save_interval", None)   # None=不定期保存
    save_best = train_cfg.get("save_best", True)
    for epoch in range(start_epoch, epochs):
        train_loss, train_recon_loss, train_anchor_loss, \
        train_bdsp_loss, train_smooth_loss, train_self_recon_loss, lr = train_one_epoch(
            model=model, optimizer=optimizer, data_loader=train_loader,
            lr_scheduler=lr_scheduler, device=device, epoch=epoch,
            loss_cfg=loss_cfg)

        val_loss, val_recon_loss, val_anchor_loss, \
        val_bdsp_loss, val_smooth_loss, val_self_recon_loss = evaluate(
            model=model, data_loader=val_loader, device=device,
            epoch=epoch, lr=lr, filefold_path=file_img_path,
            loss_cfg=loss_cfg)

        tb_writer.add_scalar("train/total_loss", train_loss, epoch)
        tb_writer.add_scalar("train/recon_loss", train_recon_loss, epoch)
        tb_writer.add_scalar("train/anchor_loss", train_anchor_loss, epoch)
        tb_writer.add_scalar("train/bdsp_loss", train_bdsp_loss, epoch)
        tb_writer.add_scalar("train/smooth_loss", train_smooth_loss, epoch)
        tb_writer.add_scalar("train/self_recon_loss", train_self_recon_loss, epoch)

        tb_writer.add_scalar("val/total_loss", val_loss, epoch)
        tb_writer.add_scalar("val/recon_loss", val_recon_loss, epoch)
        tb_writer.add_scalar("val/anchor_loss", val_anchor_loss, epoch)
        tb_writer.add_scalar("val/bdsp_loss", val_bdsp_loss, epoch)
        tb_writer.add_scalar("val/smooth_loss", val_smooth_loss, epoch)
        tb_writer.add_scalar("val/self_recon_loss", val_self_recon_loss, epoch)

        # 保存最佳模型（由 save_best 开关控制，默认开启）
        if save_best and val_loss < best_valloss:
            best_valloss = val_loss
            best_save_path = os.path.join(file_weights_path, "best_model.pth")
            torch.save({
                "model": model.module.state_dict() if use_dp else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": cfg
            }, best_save_path)
            print(f"Saved best model at epoch {epoch} with val_loss: {val_loss:.4f}")

        # 定期保存 checkpoint（save_interval 为 null 时跳过）
        if save_interval is not None and epoch % save_interval == 0:
            save_file = {
                "model": model.module.state_dict() if use_dp else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "config": cfg
            }
            torch.save(save_file, os.path.join(file_weights_path, f"checkpoint_epoch{epoch}.pth"))

    tb_writer.close()
    print(f"\nTraining completed. Best validation loss: {best_valloss:.4f}")
    print(f"Results saved to: {filefold_path}")


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
