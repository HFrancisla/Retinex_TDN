"""
test.py

Retinex 分解推理与跨分量重组评估。

加载训练好的 RetinexPointRaw 对 low/high 图像分别分解为 R 和 L，
再交叉组合（LLxLR、HLxHR 等）保存结果图，用于评估分解的解耦质量。
"""

import torch
from PIL import Image
from torchvision import transforms
from models import RetinexPointRaw, RetinexPixelClassic, RetinexPixelTrans, RetinexPixelTransMinus
import numpy as np
import cv2
import os
import argparse
import random


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.ToTensor()])

    root = "datasets/LOLv2/Test/low"
    root_high = "datasets/LOLv2/Test/high"
    assert os.path.exists(root), "file: '{}' dose not exist.".format(root)
    assert os.path.exists(root_high), "file: '{}' dose not exist.".format(root_high)

    images_path = loadfiles(root=root)
    for index in range(len(images_path)):
        assert os.path.exists(images_path[index]), "file: '{}' dose not exist.".format(images_path[index])
    print("path checking complete!")
    print("confirmly find {} images for computing".format(len(images_path)))

    images_path_high = loadfiles(root=root_high)
    for index in range(len(images_path_high)):
        assert os.path.exists(images_path_high[index]), "file: '{}' dose not exist.".format(images_path_high[index])
    print("path checking complete!")
    print("confirmly find {} images for computing".format(len(images_path_high)))

    model_cls = {
        "RetinexPointRaw": RetinexPointRaw,
        "RetinexPixelClassic": RetinexPixelClassic,
        "RetinexPixelTrans": RetinexPixelTrans,
        "RetinexPixelTransMinus": RetinexPixelTransMinus,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="RetinexPointRaw", choices=model_cls.keys())
    args = parser.parse_args()
    model = model_cls[args.model]().to(device)
    model_weight_path = r"E:\XUD\Diff-Retinex-main\model\retinexLR-test3max\experiments\LOL_loss_1_1_0.001_test3_20260611-154543\weights\checkpoint_Diff_TDN_100.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
    model.eval()

    ### LR1, LL1
    img1 = Image.open(images_path[0])
    img1 = resize(img1)
    img1 = data_transform(img1)
    img1 = img1.unsqueeze(0)
    with torch.no_grad():
        LR1, LL1 = (model(img1.to(device)))
    LR1 = LR1.squeeze(0).detach().cpu().numpy()
    LL1 = torch.cat([LL1, LL1, LL1], dim=1)
    LL1 = LL1.squeeze(0).detach().cpu().numpy()
    LR1 = np.transpose(LR1, (1, 2, 0))
    LL1 = np.transpose(LL1, (1, 2, 0))

    ### HR1, HL1
    img_high1 = Image.open(images_path_high[0])
    img_high1 = resize(img_high1)
    img_high1 = data_transform(img_high1)
    img_high1 = img_high1.unsqueeze(0)
    with torch.no_grad():
        HR1, HL1 = (model(img_high1.to(device)))
    HR1 = HR1.squeeze(0).detach().cpu().numpy()
    HL1 = torch.cat([HL1, HL1, HL1], dim=1)
    HL1 = HL1.squeeze(0).detach().cpu().numpy()
    HR1 = np.transpose(HR1, (1, 2, 0))
    HL1 = np.transpose(HL1, (1, 2, 0))

    list_LL = []
    list_LR = []
    for img_path in images_path:
        img = Image.open(img_path)
        img = resize(img)
        img = data_transform(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            LR, LL = (model(img.to(device)))
        LR = LR.squeeze(0).detach().cpu().numpy()
        LL = torch.cat([LL, LL, LL], dim=1)
        LL = LL.squeeze(0).detach().cpu().numpy()
        LR = np.transpose(LR, (1, 2, 0))
        LL = np.transpose(LL, (1, 2, 0))
        list_LL.append(LL)
        list_LR.append(LR)
        name = getnameindex(img_path)
        savepic(LR, name, flag="LR")
        savepic(LL, name, flag="LL")

        HL1xLR = HL1 * LR
        savepic(HL1xLR, name, flag="HL1xLR")

        LLxLR = LL * LR
        savepic(LLxLR, name, flag="LLxLR")

    i = 0
    for img_path in images_path_high:
        img = Image.open(img_path)
        img = resize(img)
        img = data_transform(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            HR, HL = (model(img.to(device)))
        HR = HR.squeeze(0).detach().cpu().numpy()
        HL = torch.cat([HL, HL, HL], dim=1)
        HL = HL.squeeze(0).detach().cpu().numpy()
        HR = np.transpose(HR, (1, 2, 0))
        HL = np.transpose(HL, (1, 2, 0))
        name = getnameindex(img_path)
        savepic(HR, name, flag="HR")
        savepic(HL, name, flag="HL")

        HLxHR = HL * HR
        savepic(HLxHR, name, flag="HLxHR")

        LL1xHR = LL1 * HR
        savepic(LL1xHR, name, flag="LL1xHR")

        HLxLR1 = LR1 * HL
        savepic(HLxLR1, name, flag="HLxLR1")

        HLxLR = HL * list_LR[i]
        savepic(HLxLR, name, flag="HLxLR")

        LLxHR = HR * list_LL[i]
        savepic(LLxHR, name, flag="LLxHR")
        i = i + 1


def savepic(outputpic, name, flag):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = normalize_minmax(outputpic)
    outputpic = outputpic[:, :, ::-1]

    root = "./results/LOL_v2_eval"
    root_path = os.path.join(root, flag)

    if os.path.exists("./results") is False:
        os.makedirs("./results")
    if os.path.exists(root) is False:
        os.makedirs(root)
    if os.path.exists(root_path) is False:
        os.makedirs(root_path)
    path = root_path + "/{}.png".format(name)
    cv2.imwrite(path, outputpic)
    assert os.path.exists(path), "file: '{}' dose not exist.".format(path)
    print("complete compute {}.png and save".format(name))


def normalize_minmax(img, target_min=0, target_max=255):
    img = img * (target_max - target_min) + target_min
    return img.astype(np.uint8)


def loadfiles(root):
    images_path = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"]
    images = [os.path.join(root, i) for i in os.listdir(root)
              if os.path.splitext(i)[-1] in supported]
    for index in range(len(images)):
        img_path = images[index]
        images_path.append(img_path)
    print("find {} images for computing.".format(len(images_path)))
    return images_path


def loadfiles_random(root):
    images_path = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"]
    images = [os.path.join(root, i) for i in os.listdir(root)
              if os.path.splitext(i)[-1] in supported]
    random_indices = random.sample(range(len(images)), len(images))
    for index in random_indices:
        img_path = images[index]
        images_path.append(img_path)
    print("find {} images for computing.".format(len(images_path)))
    return images_path


def getnameindex(path):
    assert os.path.exists(path), "file: '{}' dose not exist.".format(path)
    path = path.replace("\\", "/")
    label = path.split("/")[-1].split(".")[0]
    return label


def resize(image):
    original_width, original_height = image.size
    new_width = original_width - (original_width % 8)
    new_height = original_height - (original_height % 8)
    resized_image = image.resize((new_width, new_height))
    return resized_image


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
