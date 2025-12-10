import os
import torch
import torchvision
import random
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
# 定义函数 transform_augment，接受三个参数：img_list（图像列表），split（数据集的分割，比如训练集或验证集，默认为'val'），min_max（像素值的范围，默认为(0, 1)）。
# transform_augment 函数的主要作用是对输入的图像列表进行预处理和数据增强。
# 在训练阶段，它会对图像进行水平翻转以增加数据集的多样性。
# 此外，它还根据 min_max 参数对图像的像素值进行缩放和偏移，以适应不同的数值范围。
def transform_augment(img_list, split='val', min_max=(0, 1)):
    # 对 img_list 中的每个图像应用 totensor 函数,将像素值从 0-255 缩放到 0-1。
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        # 将图像张量列表 imgs 沿着第0维（批次维）堆叠成一个多维张量。
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    # 对每个图像张量进行缩放和偏移。如果 min_max 是 (0, 1)，则此步骤实际上不改变图像的像素值。但如果 min_max 是 (0, 255)，则此步骤会将像素值从 0-1 缩放回 0-255。
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    # 返回变换和增强后的图像列表。
    return ret_img
