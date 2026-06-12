import torch
from PIL import Image
from torchvision import transforms
from TDN_network import DecomNet as create_model
import numpy as np
import cv2
import os
import random

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.ToTensor()])

    root = r"D:\Datasets\NTM609\test\low"
    root_high = r"D:\Datasets\NTM609\test\high"
    assert os.path.exists(root), "file: '{}' dose not exist.".format(root)
    assert os.path.exists(root_high), "file: '{}' dose not exist.".format(root_high)

    images_path=loadfiles(root=root)
    for index in range(len(images_path)):
        assert os.path.exists(images_path[index]), "file: '{}' dose not exist.".format(images_path[index])
    print("path checking complete!")
    print("confirmly find {} images for computing".format(len(images_path)))

    images_path_high = loadfiles(root=root_high)
    for index in range(len(images_path_high)):
        assert os.path.exists(images_path_high[index]), "file: '{}' dose not exist.".format(images_path_high[index])
    print("path checking complete!")
    print("confirmly find {} images for computing".format(len(images_path_high)))

    # model = create_model().to(device)
    # model_weight_path = r"E:\XUD\Diff-Retinex-main\model\Diff_TDN-Unet-dot-unpair-NEW\experiments\TDN_train_20260212-014159\weights\checkpoint_Diff_TDN.pth"
    # model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
    # model.eval()

    for i, img_path in zip(images_path_high, images_path):
        img = Image.open(img_path)
        img = resize(img)
        img = data_transform(img)
        img = img.unsqueeze(0)

        name=getnameindex(i)
        high_GRI = normalize_grad(torch.log(img * torch.tensor(255.0) + 1))
        ###save LLxLR
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
        name=getnameindex(img_path)
        savepic(low_GRI, name, flag="low_GRI")

def normalize_grad(gradient_orig):
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm

def savepic(outputpic, name, flag):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    # outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic = normalize_minmax(outputpic)
    outputpic=outputpic[:, :, ::-1]

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
    src_min, src_max = img.min(), img.max()
    # if src_max == src_min:
    img = img * (target_max - target_min) + target_min
    return img.astype(np.uint8)

    # normalized = (img - src_min) / (src_max - src_min)  # [0, 1]
    # normalized = normalized * (target_max - target_min) + target_min
    # return normalized.astype(np.uint8)

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