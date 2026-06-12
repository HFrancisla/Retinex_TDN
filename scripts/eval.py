"""
eval.py

对数域梯度图评估工具。

对 low/high 图像分别进行 log/exp 变换后计算归一化梯度图（GRI），
用于辅助分析光照与反射分量的频域特性。
"""

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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

    for i, img_path in zip(images_path_high, images_path):
        img = Image.open(img_path)
        img = resize(img)
        img = data_transform(img)
        img = img.unsqueeze(0)

        name = getnameindex(i)
        high_GRI = normalize_grad(torch.log(img * torch.tensor(255.0) + 1))
        high_GRI = high_GRI.squeeze(0).detach().cpu().numpy()
        high_GRI = np.transpose(high_GRI, (1, 2, 0))

        savepic(high_GRI, name, flag="high_GRI")

    for img_path in images_path_high:
        img = Image.open(img_path)
        img = resize(img)
        img = data_transform(img)
        img = img.unsqueeze(0)
        low_GRI = normalize_grad(torch.exp(img * (torch.log(torch.tensor(255.0)))))

        low_GRI = low_GRI.squeeze(0).detach().cpu().numpy()
        low_GRI = np.transpose(low_GRI, (1, 2, 0))
        name = getnameindex(img_path)
        savepic(low_GRI, name, flag="low_GRI")


def normalize_grad(gradient_orig):
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = (gradient_orig - grad_min) / (grad_max - grad_min + 1e-4)
    return grad_norm


def savepic(outputpic, name, flag):
    outputpic = np.clip(outputpic, 0.0, 1.0)
    outputpic = normalize_minmax(outputpic)
    outputpic = outputpic[:, :, ::-1]

    root = "./results/LOL_high_eval"
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
    images = sorted([os.path.join(root, i) for i in os.listdir(root)
                     if os.path.splitext(i)[-1] in supported])
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
