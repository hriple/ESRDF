import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np



#直方图分布Loss
def _soft_histogram(x, bins, sigma, eps):
    """
    可微直方图生成核心函数
    """
    batch, channels, height, width = x.shape
    x = x.view(batch, channels, -1)  # [B, C, H*W]
    x = x.unsqueeze(-1)  # [B, C, H*W, 1]
    
    # 动态生成bin中心点（根据输入设备）
    centers = torch.linspace(0, 1, bins, device=x.device).view(1, 1, 1, -1)  # [1,1,1,bins]
    
    # 高斯核权重计算 (可微分)
    diff = (x - centers) / sigma
    sqrt_2pi = torch.sqrt(2 * torch.tensor(torch.pi, device=x.device))
    weights = torch.exp(-0.5 * diff**2) / (sigma * sqrt_2pi)
    
    # 概率归一化
    hist = torch.sum(weights, dim=2)  # [B, C, bins]
    hist = hist / hist.sum(dim=2, keepdim=True)  # 概率归一化
    return hist

def histogram_loss(input, target, bins=256, sigma=0.01, eps=1e-10):
    """
    函数式直方图损失实现（KL散度版本）
    
    参数:
        input (Tensor): 生成图像 [B, C, H, W]
        target (Tensor): 目标图像 [B, C, H, W]
        bins (int): 直方图分bin数量
        sigma (float): 高斯核带宽
        eps (float): 数值稳定性常数
        
    返回:
        loss (Tensor): 标量损失值
    """
    # 确保输入范围在 [0, 1]
    input = input.clamp(0, 1)
    target = target.clamp(0, 1)
    
    # 生成可微直方图
    hist_input = _soft_histogram(input, bins, sigma, eps)
    hist_target = _soft_histogram(target, bins, sigma, eps)
    
    #两种直方图Loss, 一种是 KL散度，不对称，计算量大，直方图更平滑；另一种计算重叠部分，计算量小，对称；
    # 计算KL散度 (反向传播友好)
    kl_div = hist_target * (torch.log(hist_target + eps) - torch.log(hist_input + eps))
    kl_loss = kl_div.sum(dim=1).mean()  # 或者 kl_div.mean(dim=(1, 2)).mean()
    
    return kl_loss


def content_mse_loss(hr, lr, vgg16, deep_num = 0):
    loss_mse = nn.MSELoss()
    hr_features = vgg16(hr)
    output_features = vgg16(lr)
    hr_recon = hr_features[deep_num]      
    output_recon = output_features[deep_num]
    content_loss = loss_mse(hr_recon, output_recon)
    return content_loss


# 实例化一次 L1Loss
mae_loss_fn = nn.L1Loss()
def mae_loss(img1, img2):
    return mae_loss_fn(img1,img2)

# # 实例化一次 L2Loss（MSELoss）
mse_loss_fn = nn.MSELoss()
def mse_loss(img1, img2):
    return mse_loss_fn(img1, img2)

import torch
import torch.nn.functional as F

def calculate_tensor_entropy(images_tensor):
    """
    计算输入tensor图片的熵值
    
    参数:
    images_tensor (torch.Tensor): 输入的图片tensor，范围[-1, 1]，形状为[batch_size, channels, height, width]
    
    返回:
    torch.Tensor: 每张图片的熵值，形状为[batch_size]
    """
    # 确保输入是有效的tensor
    if not isinstance(images_tensor, torch.Tensor):
        images_tensor = torch.tensor(images_tensor)
    
    # 将像素值从[-1, 1]范围转换为[0, 1]范围
    normalized_tensor = (images_tensor + 1) / 2
    
    # 将像素值离散化为256个bins，模拟图像像素的分布
    num_bins = 256
    batch_size = normalized_tensor.size(0)
    entropy_values = torch.zeros(batch_size)
    
    for i in range(batch_size):
        # 对每个样本计算直方图
        hist = torch.histc(normalized_tensor[i], bins=num_bins, min=0, max=1)
        
        # 计算概率分布
        probs = hist / hist.sum()
        
        # 过滤掉概率为0的bin，避免计算log(0)
        probs = probs[probs > 0]
        
        # 计算熵值
        entropy = -torch.sum(probs * torch.log2(probs))
        
        # 归一化熵值到[0, 8]范围（8位图像的最大熵）
        normalized_entropy = entropy / 8.0
        
        entropy_values[i] = normalized_entropy
    
    return entropy_values.mean()

def block_entropy(images_tensor, block_size=8):
    """
    计算输入tensor图片的分块熵值
    
    参数:
    images_tensor (torch.Tensor): 输入的图片tensor，范围[-1, 1]，形状为[batch_size, channels, height, width]
    block_size (int): 分块大小，默认为8
    
    返回:
    torch.Tensor: 每张图片的每个分块的熵值，形状为[batch_size, num_blocks_h, num_blocks_w]
    """
    # 确保输入是有效的tensor
    if not isinstance(images_tensor, torch.Tensor):
        images_tensor = torch.tensor(images_tensor)
    
    # 将像素值从[-1, 1]范围转换为[0, 1]范围
    normalized_tensor = (images_tensor + 1) / 2
    
    batch_size, channels, height, width = normalized_tensor.size()
    
    # 计算分块数量
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size
    
    # 初始化熵值tensor
    block_entropies = torch.zeros(batch_size, num_blocks_h, num_blocks_w)
    
    # 遍历每个分块
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # 提取当前分块
            block = normalized_tensor[
                :, :, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size
            ]
            
            # 将分块展平为一维向量 (修改点：使用reshape而不是view)
            block_flat = block.reshape(batch_size, -1)
            
            # 对每个样本的分块计算直方图
            num_bins = 256
            batch_entropies = torch.zeros(batch_size)
            
            for b in range(batch_size):
                hist = torch.histc(block_flat[b], bins=num_bins, min=0, max=1)
                
                # 计算概率分布
                probs = hist / hist.sum()
                
                # 过滤掉概率为0的bin
                probs = probs[probs > 0]
                
                # 计算熵值
                if len(probs) > 0:
                    entropy = -torch.sum(probs * torch.log2(probs))
                    # 归一化到[0,1]
                    batch_entropies[b] = entropy / 8.0
            
            block_entropies[:, i, j] = batch_entropies
    
    return block_entropies.mean()

def image_compare_loss(x, y, device, vgg16):
    device = device
    l1_loss = mae_loss(x, y)
    l2_loss = mse_loss(x, y)
    hist_loss = histogram_loss(x, y, bins=256)
    content_loss = content_mse_loss(x, y, vgg16)
    # 分块计算 熵
    y_entropy = block_entropy(y).to(device)
    # print("----l1_loss:",l1_loss.item())
    # print("----hist_loss:",hist_loss.item())
    # print("----content_loss:",content_loss.item())
    # print("-----")
    # return 10 * l1_loss, 100 * hist_loss, content_loss
    return 1 * l1_loss, 100 * hist_loss, content_loss
