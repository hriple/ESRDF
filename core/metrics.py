import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
from PIL import Image
import torch

import torch
from lpips import LPIPS

def calculate_lpips(imgA, imgB, net='vgg'):
    model = LPIPS(net=net)
    device = next(model.parameters()).device
    tA = t(imgA).to(device)
    tB = t(imgB).to(device)
    dist01 = model.forward(tA, tB).item()
    return max(0, min(1, dist01))
    # return dist01
    

def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)
    
	# 确保图像像素值在[0, 255]范围内
    img = np.clip(img, 0, 255)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


from pytorch_fid import fid_score
import shutil

def calculate_fid(hr_folder, sr_folder, device='cuda:0', batch_size=32, dims=2048):
    """计算FID指标"""
    return fid_score.calculate_fid_given_paths(
        [hr_folder, sr_folder],
        batch_size=batch_size,
        device=device,  # 直接使用字符串设备标识
        dims=dims
    )



def add_images_and_normalize(sr_img, inf_img):
    # sr_img 是-1到 1 ,inf_img 0到1
    # 将PIL图像转换为NumPy数组
    sr_array = np.array(sr_img)
    inf_array = np.array(inf_img)
    # 确保两个图像大小相同
    if sr_array.shape != inf_array.shape:
        raise ValueError("The size of two images must be the same.")
    result_array = sr_array + inf_array
    result_array = (result_array / 2.0 + 0.5) * 255
    result_array = result_array.clip(0, 255)  # 确保数值在0到255之间
    result_array = result_array.astype('uint8')  # 转换为整数类型
    result_array = np.transpose(result_array, (1, 2, 0))
    # 将结果数组转换为PIL图像
    result_img = Image.fromarray(result_array)
    # 转换回0-255范围
    return result_img, result_array


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    # print('tensor',tensor)
    # print('tensor',tensor.shape)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
