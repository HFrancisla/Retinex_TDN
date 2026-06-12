"""
train.py

Retinex 分解模型训练入口。

解析命令行参数，加载配对数据集，构建 DecomNet 及优化器，
执行训练-验证循环并保存权重、TensorBoard 日志与中间可视化结果。

用法示例：
    python train.py --data-path /path/to/dataset --epochs 300 --batch-size 2 --lr 0.0001
"""

import os
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from data import MyDataSet, transforms as T
from models import DecomNet
from utils import read_data, train_one_epoch, evaluate, create_lr_scheduler
import datetime


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./experiments") is False:
        os.makedirs("./experiments")

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "./experiments/LOL_loss_1_1_0.001_test3_{}".format(file_name)
    os.makedirs(filefold_path)
    file_img_path = os.path.join(filefold_path, "img")
    os.makedirs(file_img_path)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path)

    tb_writer = SummaryWriter(log_dir=file_log_path)

    best_valloss = 1e5
    start_epoch = 0

    train_low_path, train_high_path, val_low_path, val_high_path = read_data(args.data_path)

    data_transform = {
        "train": T.Compose([T.RandomCrop(256),
                            T.RandomHorizontalFlip(0.5),
                            T.RandomVerticalFlip(0.5),
                            T.ToTensor()]),

        "val": T.Compose([T.ToTensor()])}

    train_dataset = MyDataSet(images_low_path=train_low_path,
                              images_high_path=train_high_path,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_low_path=val_low_path,
                            images_high_path=val_high_path,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)

    model = DecomNet().to(device)
    if args.use_dp == True:
        model = torch.nn.DataParallel(model).cuda()

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        print(model.load_state_dict(weights_dict, strict=False))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5E-5)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    save_epoch = 20
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_rec_loss, train_equal_R_loss, \
        train_smooth_high_loss, lr = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                lr_scheduler=lr_scheduler,
                                                device=device,
                                                epoch=epoch)

        val_loss, val_rec_loss, val_equal_R_loss, \
        val_smooth_high_loss = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch, lr=lr, filefold_path=file_img_path)

        tb_writer.add_scalar("train_total_loss", train_loss, epoch)
        tb_writer.add_scalar("train_rec_loss", train_rec_loss, epoch)
        tb_writer.add_scalar("train_equal_R_loss", train_equal_R_loss, epoch)
        tb_writer.add_scalar("train_smooth_high_loss", train_smooth_high_loss, epoch)

        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_rec_loss", val_rec_loss, epoch)
        tb_writer.add_scalar("val_equal_R_loss", val_equal_R_loss, epoch)
        tb_writer.add_scalar("val_smooth_high_loss", val_smooth_high_loss, epoch)

        if epoch % save_epoch == 0:
            if args.use_dp == True:
                save_file = {"model": model.module.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            else:
                save_file = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            torch.save(save_file, file_weights_path + "/" + "checkpoint_Diff_TDN_{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data-path', type=str,
                        default=r"E:\XUD\datasets\NTM")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', default=False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--gpu_id', default='2', help='device id (i.e. 0, 1, 2 or 3)')
    opt = parser.parse_args()

    main(opt)
