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
import shutil
import argparse
import datetime
import time
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data import MyDataSet, UnpairedDataSet, PureLowDataSet, PureLowSingleDataSet, transforms as T
from models import RetinexPointRaw, RetinexPixelClassic, RetinexPixelTrans, RetinexPixelTransMinus
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

    def isatty(self):
        """透传终端能力，供 tqdm 判断是否适合原地刷新。"""
        return self._stream.isatty()

    def fileno(self):
        """透传文件描述符，供 tqdm 获取动态终端宽度。"""
        return self._stream.fileno()

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

    # mode: 只取 _pixel / _point 后缀（data_mode 已在目录路径中）
    full_mode = loss_cfg.get("mode", "unknown")
    if full_mode.endswith('_pixel'):
        mode = 'pixel'
    elif full_mode.endswith('_point'):
        mode = 'point'
    else:
        mode = full_mode  # 兜底

    # 损失权重：显式声明的全部输出，方便横向对比
    parts = []
    for key, abbr in LOSS_ABBR:
        if key in loss_cfg:
            val = loss_cfg[key]
            # 整数值自动补 .0 (1→1.0, 0→0.0)
            if isinstance(val, (int, float)) and float(val).is_integer():
                val = f"{float(val):.1f}"
            if key == 'anchor_weight' and 'anchor_version' in loss_cfg:
                parts.append(f"{val}{abbr}{loss_cfg['anchor_version']}")
            else:
                parts.append(f"{val}{abbr}")

    return "_".join([dataset, mode] + parts)


def validate_pipeline_config(cfg):
    """校验 data/loss/model 三者语义及关键训练参数。"""
    data_cfg = cfg.get('data', {})
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('training', {})
    loss_cfg = cfg.get('loss', {})

    data_mode = data_cfg.get('mode')
    valid_data_modes = {'paired', 'unpaired', 'pure_low_single', 'pure_low_double'}
    if data_mode not in valid_data_modes:
        raise ValueError(f"invalid data.mode={data_mode!r}; choose from {sorted(valid_data_modes)}")

    loss_mode = loss_cfg.get('mode', '')
    if not loss_mode.startswith(f'{data_mode}_'):
        raise ValueError(
            f"data.mode={data_mode!r} is inconsistent with loss.mode={loss_mode!r}"
        )

    model_name = model_cfg.get('name', 'RetinexPointRaw')
    valid_models = {
        'RetinexPointRaw', 'RetinexPixelClassic',
        'RetinexPixelTrans', 'RetinexPixelTransMinus',
    }
    if model_name not in valid_models:
        raise ValueError(f'invalid model.name={model_name!r}; choose from {sorted(valid_models)}')
    expected_l_type = 'point' if model_name == 'RetinexPointRaw' else 'pixel'
    if not loss_mode.endswith(f'_{expected_l_type}'):
        raise ValueError(
            f"model.name={model_name!r} requires a _{expected_l_type} loss, "
            f"got loss.mode={loss_mode!r}"
        )

    for name in ('crop_size', 'batch_size', 'val_batch_size'):
        value = data_cfg.get(name, 1)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f'data.{name} must be a positive integer, got {value!r}')
    num_workers = data_cfg.get('num_workers', 0)
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(f'data.num_workers must be a non-negative integer, got {num_workers!r}')
    if train_cfg.get('max_iterations', 0) < 0:
        raise ValueError('training.max_iterations must be >= 0')
    if float(train_cfg.get('lr', 1e-4)) <= 0:
        raise ValueError('training.lr must be > 0')
    for key, value in loss_cfg.items():
        if key.endswith('_weight'):
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f'loss.{key} must be a finite non-negative number, got {value!r}')

    # 先执行 loss 字段完整性检查，避免创建实验目录后才失败。
    _build_loss_function(loss_cfg)


def get_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


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
                    "save_best_ckpt": True,
                    "keep_top_ckpt": 2,
                    "max_save_images": 250
                }
            },
            "loss": {},
            "resume": {
                "checkpoint": args.resume,
                "weights": args.weights
            },
            "device": args.device
        }
        if args.mode == 'paired':
            cfg['loss'] = {
                'mode': 'paired_point',
                'recon_weight_high': 1.0,
                'recon_weight_low': 0.3,
                'cross_recon_weight_low': 0.001,
                'cross_recon_weight_high': 0.001,
                'equal_r_weight': 0.1,
            }
        elif args.mode == 'unpaired':
            cfg['loss'] = {
                'mode': 'unpaired_point', 'recon_weight': 1.0,
                'anchor_weight': 0.05, 'bdsp_weight': 0.05,
                'self_recon_weight': 0.05, 'anchor_version': 'v2',
            }
        elif args.mode == 'pure_low_double':
            cfg['loss'] = {
                'mode': 'pure_low_double_point', 'recon_weight': 1.0,
                'anchor_weight': 0.05, 'bdsp_weight': 0.05,
                'self_recon_weight': 0.05, 'reflect_weight': 0.1,
                'anchor_version': 'v2',
            }
        else:
            cfg['loss'] = {
                'mode': 'pure_low_single_point', 'recon_weight': 1.0,
                'anchor_weight': 0.05, 'bdsp_weight': 0.05,
                'anchor_version': 'v2',
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

    validate_pipeline_config(cfg)

    # 必须在首次 CUDA API 调用前设置可见设备。
    if str(device_str).startswith('cuda'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(model_cfg.get("gpu_id", "0"))

    # ---- 随机种子（可复现）----
    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

    # ---- 设备设置 ----
    if str(device_str).startswith('cuda') and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(device_str)

    # ---- 生成实验目录名 ----
    experiment_name = generate_experiment_name(cfg)
    model_name = model_cfg.get("name", "RetinexPointRaw")
    data_mode = data_cfg.get("mode", "unknown")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = resume_cfg.get("checkpoint", "")
    if checkpoint_path:
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint_path}")
        checkpoint_parent = os.path.dirname(checkpoint_path)
        if os.path.basename(checkpoint_parent) != 'weights':
            raise ValueError("resume checkpoint must be located in an experiment's weights directory")
        filefold_path = os.path.dirname(checkpoint_parent)
        print(f"Resume in existing experiment directory: {filefold_path}")
    else:
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
    config_filename = f"config_resume_{timestamp}.yaml" if checkpoint_path else "config.yaml"
    config_save_path = os.path.join(filefold_path, config_filename)
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"Config saved to: {config_save_path}")

    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Directory:  {filefold_path}")
    print(f"{'='*60}\n")

    # ---- TensorBoard ----
    tb_writer = SummaryWriter(log_dir=file_log_path)

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
        from data.transforms import RandomGamma, RandomBrightness
        data_transform = {
            "train": T.Compose([T.RandomCrop(crop_size),
                                T.RandomHorizontalFlip(0.5),
                                T.RandomVerticalFlip(0.5),
                                T.ToTensor()]),
            "val": T.Compose([T.ToTensor()])
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
            "train": torchvision_T.Compose([torchvision_T.RandomCrop(crop_size, pad_if_needed=True),
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
            "train": torchvision_T.Compose([torchvision_T.RandomCrop(crop_size, pad_if_needed=True),
                                              torchvision_T.RandomHorizontalFlip(0.5),
                                              torchvision_T.RandomVerticalFlip(0.5),
                                              torchvision_T.ToTensor()]),
            "val": torchvision_T.Compose([torchvision_T.ToTensor()])
        }
        train_dataset = UnpairedDataSet(images_low_path=train_low_path,
                                        images_high_path=train_high_path,
                                        transform=data_transform["train"],
                                        random_pair=True)
        val_dataset = UnpairedDataSet(images_low_path=val_low_path,
                                      images_high_path=val_high_path,
                                      transform=data_transform["val"],
                                      random_pair=False)

    batch_size = data_cfg.get("batch_size", 2)
    val_batch_size = data_cfg.get("val_batch_size", 1)
    num_workers = data_cfg.get("num_workers", 0)
    print(f'Using {num_workers} dataloader workers for train_loader (batch_size={batch_size})')
    print(f'val_loader: batch_size={val_batch_size}, workers=0')

    train_generator = torch.Generator()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=device.type == 'cuda',
                                               num_workers=num_workers,
                                               generator=train_generator,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=val_batch_size,
                                             shuffle=False,
                                             pin_memory=device.type == 'cuda',
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
    elif model_name == "RetinexPixelTransMinus":
        model = RetinexPixelTransMinus(
            dim=model_cfg.get("dim", 24),
            l_heads=model_cfg.get("l_heads", 1),
            l_ffn_expansion=model_cfg.get("l_ffn_expansion", 2.66),
        ).to(device)
    elif model_name == "RetinexPointRaw":
        model = RetinexPointRaw(dim=model_cfg.get("dim", 24)).to(device)
    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Supported: RetinexPointRaw, RetinexPixelClassic, RetinexPixelTrans, RetinexPixelTransMinus. "
            f"Please check your config YAML → model.name."
        )
    use_dp = model_cfg.get("use_dp", False)
    if use_dp:
        if device.type != 'cuda':
            raise ValueError('model.use_dp=true requires a CUDA device')
        model = torch.nn.DataParallel(model).to(device)

    # ---- 加载权重 ----
    weights_path = resume_cfg.get("weights", "")
    if weights_path:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"weights file does not exist: {weights_path}")
        # 路径由用户显式提供；完整训练 checkpoint 含 optimizer/RNG 等非权重对象。
        weights_file = torch.load(weights_path, map_location=device, weights_only=False)
        weights_dict = weights_file.get("model", weights_file)
        load_target = model.module if use_dp else model
        strict_weights = resume_cfg.get('strict_weights', True)
        print(load_target.load_state_dict(weights_dict, strict=strict_weights))

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
    if not isinstance(warmup_iterations, int) or not 0 <= warmup_iterations <= max_iterations:
        raise ValueError(
            f'warmup_iterations must be an integer in [0, {max_iterations}], '
            f'got {warmup_iterations!r}'
        )
    lr_scheduler = create_lr_scheduler(optimizer, max_iterations, warmup_iterations)

    # ---- 恢复训练 ----
    start_iter = 0
    eval_cfg = train_cfg.get("eval", {})
    selection_metric = eval_cfg.get('selection_metric', 'total_loss')
    selection_mode = eval_cfg.get(
        'selection_mode', 'max' if selection_metric == 'psnr' else 'min'
    )
    if selection_mode not in ('min', 'max'):
        raise ValueError("training.eval.selection_mode must be 'min' or 'max'")
    best_metric_value = float('inf') if selection_mode == 'min' else -float('inf')
    last_val_metrics = None
    last_val_psnr = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_training = checkpoint.get('config', {}).get('training', {})
        saved_max_iterations = saved_training.get('max_iterations')
        if saved_max_iterations not in (None, max_iterations):
            raise ValueError(
                f'resume must keep the original max_iterations={saved_max_iterations}; '
                f'got {max_iterations}. Use resume.weights to start a new fine-tuning schedule.'
            )
        if checkpoint.get('selection_metric', selection_metric) != selection_metric:
            raise ValueError('resume checkpoint selection_metric differs from current config')
        if checkpoint.get('selection_mode', selection_mode) != selection_mode:
            raise ValueError('resume checkpoint selection_mode differs from current config')
        load_target = model.module if use_dp else model
        load_target.load_state_dict(checkpoint['model'])
        if 'optimizer' not in checkpoint:
            raise KeyError('resume checkpoint has no optimizer state; use resume.weights for fine-tuning')
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_iter = checkpoint.get('global_iter', checkpoint.get('epoch', 0) * iterations_per_epoch)
        best_metric_value = checkpoint.get('best_metric_value', best_metric_value)
        last_val_metrics = checkpoint.get('last_val_metrics')
        last_val_psnr = checkpoint.get('last_val_psnr')
        set_rng_state(checkpoint.get('rng_state'))
        if start_iter > max_iterations:
            raise ValueError(
                f'checkpoint global_iter={start_iter} exceeds max_iterations={max_iterations}'
            )
        print(f"Resumed from checkpoint: step {start_iter}")

    # ---- 评估/保存间隔 ----
    eval_interval = eval_cfg.get("eval_interval", 500)
    save_img_interval = eval_cfg.get("save_img_interval", eval_interval * 4)
    save_best_ckpt = eval_cfg.get("save_best_ckpt", True)
    save_ckpt_interval = train_cfg.get("save_ckpt_interval", 5000)
    log_interval = train_cfg.get("log_interval", 1000)

    # 磁盘清理策略
    keep_top_ckpt = eval_cfg.get("keep_top_ckpt", 2)
    max_save_images = eval_cfg.get("max_save_images", 250)

    saved_ckpt_files = checkpoint.get('saved_ckpt_files', []) if checkpoint_path else []
    saved_img_folders = checkpoint.get('saved_img_folders', []) if checkpoint_path else []

    print(f"eval_interval: {eval_interval} steps")
    print(f"log_interval: {log_interval} steps")
    print(f"save_ckpt_interval: {save_ckpt_interval} steps")
    print(f"save_img_interval: {save_img_interval} steps")
    print(f"keep_top_ckpt: {keep_top_ckpt} (0=keep all)")
    print(f"max_save_images: {max_save_images}")

    # ---- 校验 interval 约束 ----
    if log_interval <= 0:
        raise ValueError(f"log_interval must be > 0, got {log_interval}")
    if eval_interval <= 0:
        raise ValueError(f"eval_interval must be > 0, got {eval_interval}")
    if save_img_interval < 0:
        raise ValueError(f"save_img_interval must be >= 0, got {save_img_interval}")
    if save_ckpt_interval < 0:
        raise ValueError(f"save_ckpt_interval must be >= 0, got {save_ckpt_interval}")
    if not isinstance(keep_top_ckpt, int) or keep_top_ckpt < 0:
        raise ValueError(f"keep_top_ckpt must be a non-negative integer, got {keep_top_ckpt!r}")
    if not isinstance(max_save_images, int) or max_save_images < 0:
        raise ValueError(f"max_save_images must be a non-negative integer, got {max_save_images!r}")
    if save_img_interval > 0 and save_img_interval % eval_interval != 0:
        raise ValueError(
            f"save_img_interval ({save_img_interval}) must be a multiple of eval_interval ({eval_interval})"
        )
    if save_ckpt_interval > 0 and save_ckpt_interval % eval_interval != 0:
        raise ValueError(
            f"save_ckpt_interval ({save_ckpt_interval}) must be a multiple of "
            f"eval_interval ({eval_interval}) so every ranked checkpoint has a fresh metric"
        )

    # ---- 构建损失函数（复用，不重复构建）----
    loss_function = _build_loss_function(loss_cfg).to(device)

    # ---- 训练循环（step-based）----
    global_iter = start_iter
    epoch = global_iter // iterations_per_epoch
    batch_offset = global_iter % iterations_per_epoch
    log_accum = {}
    log_sample_count = 0
    eval_accum = {}
    eval_sample_count = 0
    last_eval_iter = checkpoint.get('last_eval_iter', -1) if checkpoint_path else -1
    train_start = time.perf_counter()

    def _fmt(v):
        """小值用科学计数法，避免 .4f 显示为 0.0000。"""
        if abs(v) < 0.0001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}"

    def _start_epoch_iterator(epoch_index, skip_batches=0):
        """每个 epoch 使用独立确定性种子；resume 时重放至准确 batch 游标。"""
        epoch_seed = seed + epoch_index
        random.seed(epoch_seed)
        np.random.seed(epoch_seed % (2 ** 32))
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
        train_generator.manual_seed(epoch_seed)
        iterator = iter(train_loader)
        for _ in range(skip_batches):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError('invalid resume batch offset') from exc
        return iterator

    data_iter = _start_epoch_iterator(epoch, batch_offset)

    def _accumulate(target, metrics, batch_size):
        for key, value in metrics.items():
            target[key] = target.get(key, 0.0) + value * batch_size

    def _averages(accum, count):
        return {key: value / count for key, value in accum.items()} if count else {}

    def _selection_value(metrics, psnr):
        if selection_metric == 'psnr':
            if psnr is None:
                raise ValueError('selection_metric=psnr is only valid for paired mode')
            return psnr
        if selection_metric not in metrics:
            raise ValueError(
                f"unknown selection_metric={selection_metric!r}; available={sorted(metrics)}"
            )
        return metrics[selection_metric]

    def _is_better(value, best):
        return value < best if selection_mode == 'min' else value > best

    def _worst_index(items):
        scores = [item[0] for item in items]
        return scores.index(max(scores) if selection_mode == 'min' else min(scores))

    def _checkpoint_payload(step):
        return {
            'model': model.module.state_dict() if use_dp else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'global_iter': step,
            'best_metric_value': best_metric_value,
            'selection_metric': selection_metric,
            'selection_mode': selection_mode,
            'last_val_metrics': last_val_metrics,
            'last_val_psnr': last_val_psnr,
            'last_eval_iter': last_eval_iter,
            'saved_ckpt_files': saved_ckpt_files,
            'saved_img_folders': saved_img_folders,
            'rng_state': get_rng_state(),
            'config': cfg,
        }

    def _run_evaluation(step, save_img, train_metrics):
        nonlocal best_metric_value, last_val_metrics, last_val_psnr, last_eval_iter
        val_metrics, val_psnr = evaluate(
            model=model, data_loader=val_loader, device=device,
            lr=optimizer.param_groups[0]['lr'], filefold_path=file_img_path,
            loss_function=loss_function, save_images=save_img, global_iter=step,
            max_save_images=max_save_images,
        )
        if not val_metrics:
            raise ValueError('validation dataset produced no samples')

        for key, value in train_metrics.items():
            tb_writer.add_scalar(f'train/{key}', value, step)
        for key, value in val_metrics.items():
            tb_writer.add_scalar(f'val/{key}', value, step)
        if val_psnr is not None:
            tb_writer.add_scalar('val/psnr', val_psnr, step)

        parts = [f"total: {val_metrics['total_loss']:.4f}"]
        for name in ('recon', 'cross_recon', 'anchor', 'bdsp', 'smooth',
                     'self_recon', 'equal_r', 'reflect'):
            key = f'{name}_weighted_loss'
            if key in val_metrics:
                parts.append(f'{name}(weighted): {_fmt(val_metrics[key])}')
        if val_psnr is not None:
            parts.append(f'PSNR(proxy): {val_psnr:.2f}dB')
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{now}] [eval  step {step:>6d}] " + ' | '.join(parts))

        last_val_metrics = val_metrics
        last_val_psnr = val_psnr
        last_eval_iter = step
        score = _selection_value(val_metrics, val_psnr)

        if _is_better(score, best_metric_value):
            best_metric_value = score
            if save_best_ckpt:
                best_save_path = os.path.join(file_weights_path, 'best_model.pth')
                torch.save(_checkpoint_payload(step), best_save_path)
                print(
                    f"Saved best model at step {step} | {selection_metric}: "
                    f"{score:.6f} ({selection_mode})"
                )

        if save_img and keep_top_ckpt > 0:
            img_folder = os.path.join(file_img_path, str(step))
            saved_img_folders.append((score, img_folder))
            if len(saved_img_folders) > keep_top_ckpt:
                _, worst_folder = saved_img_folders.pop(_worst_index(saved_img_folders))
                if os.path.exists(worst_folder):
                    shutil.rmtree(worst_folder)
                    print(f'[cleanup] Removed visualization folder: {worst_folder}')
        return score

    while global_iter < max_iterations:
        # 获取下一个 batch，epoch 结束时重新 shuffle
        try:
            data = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = _start_epoch_iterator(epoch)
            data = next(data_iter)

        # 单步训练
        loss_vals = train_step(model, optimizer, loss_function, data, device, lr_scheduler)
        step_batch_size = int(loss_vals.pop('_batch_size'))
        _accumulate(log_accum, loss_vals, step_batch_size)
        log_sample_count += step_batch_size
        _accumulate(eval_accum, loss_vals, step_batch_size)
        eval_sample_count += step_batch_size
        global_iter += 1
        lr = optimizer.param_groups[0]["lr"]

        # 定期打印训练进度
        if global_iter % log_interval == 0:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            avg = _averages(log_accum, log_sample_count)
            current_epoch = global_iter // iterations_per_epoch + 1  # 1-indexed
            parts = [f"ep:{current_epoch}",
                     f"total: {avg['total_loss']:.4f}"]
            for name in ('recon', 'cross_recon', 'anchor', 'bdsp', 'smooth',
                         'self_recon', 'equal_r', 'reflect'):
                key = f'{name}_weighted_loss'
                if key in avg:
                    parts.append(f'{name}(weighted): {_fmt(avg[key])}')
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
            if device.type == 'cuda':
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                parts.append(f"mem: {mem_reserved:.2f}GB")
            print(f"[{now}] [iter: {global_iter:>6d}/{max_iterations}] " + " | ".join(parts))

            # 下一条训练日志只统计新的 log_interval 个 step。
            log_accum.clear()
            log_sample_count = 0

        # ---- 评估 ----
        if global_iter % eval_interval == 0:
            print()
            save_img = save_img_interval > 0 and global_iter % save_img_interval == 0
            _run_evaluation(global_iter, save_img, _averages(eval_accum, eval_sample_count))
            eval_accum.clear()
            eval_sample_count = 0

        # ---- 定期保存 checkpoint ----
        if save_ckpt_interval > 0 and global_iter % save_ckpt_interval == 0:
            score = _selection_value(last_val_metrics, last_val_psnr)
            ckpt_name = f"checkpoint_{global_iter}_{selection_metric}{score:.6f}.pth"
            ckpt_path = os.path.join(file_weights_path, ckpt_name)
            keep_current_ckpt = True

            # ---- 清理旧 checkpoint：仅保留 val_loss 最低的 N 个（best_model.pth 不受影响）----
            if keep_top_ckpt > 0:
                saved_ckpt_files.append((score, global_iter, ckpt_path))
                if len(saved_ckpt_files) > keep_top_ckpt:
                    _, _, old_path = saved_ckpt_files.pop(_worst_index(saved_ckpt_files))
                    if old_path == ckpt_path:
                        keep_current_ckpt = False
                    elif os.path.exists(old_path):
                        os.remove(old_path)
                        print(f"[cleanup] Removed old checkpoint: {os.path.basename(old_path)}")
            if keep_current_ckpt:
                torch.save(_checkpoint_payload(global_iter), ckpt_path)
            else:
                print(f'[cleanup] Current checkpoint was outside top-{keep_top_ckpt}; not saved')

    # max_iterations 不一定命中 eval_interval，结束前必须验证最终状态。
    if last_eval_iter != global_iter:
        print()
        _run_evaluation(global_iter, False, _averages(eval_accum, eval_sample_count))

    last_save_path = os.path.join(file_weights_path, 'last_model.pth')
    torch.save(_checkpoint_payload(global_iter), last_save_path)
    print(f'Saved final model: {last_save_path}')

    print()
    tb_writer.close()
    print(
        f"\nTraining completed. Best {selection_metric} "
        f"({selection_mode}): {best_metric_value:.6f}"
    )
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
    parser.add_argument(
        "--mode", type=str, default="paired",
        choices=["paired", "unpaired", "pure_low_single", "pure_low_double"],
        help="data loading mode",
    )
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', action='store_true', help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')

    opt = parser.parse_args()
    main(opt)
