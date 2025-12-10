import sys
from calculate_metrics import calculate_fid, calculate_lpips, calculate_psnr, calculate_ssim
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime

from PixelPredictor import PixelPredictor
# from SRCNN import SRCNN
from loss import image_compare_loss
from dataset import CustomDataset
from vgg import Vgg16

from pytorch_fid import fid_score
import warnings

# 忽略所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

def save_model(model, epoch, args):
    """
    保存模型权重到对应epoch的结果文件夹内
    
    Args:
        model: 待保存的模型
        epoch: 当前训练轮次
        args: 命令行参数
    """
    # 构建保存路径：pre_model/epoch_{epoch}/model.pth
    epoch_folder = os.path.join(args.model_save, f'epoch_{epoch}')
    os.makedirs(epoch_folder, exist_ok=True)  # 确保文件夹存在
    
    # 生成文件名（固定为 model.pth，避免重复时间戳）
    model_save_path = os.path.join(epoch_folder, 'model.pth')
    
    # 保存模型状态字典
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at Epoch {epoch}: {model_save_path}")


# 验证函数（使用args.device）
def valid(model, epoch, args):
    valid_dataset = CustomDataset(args, args.valid_dir)
    valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False)

    model.eval()
    hr_folder = os.path.join(args.results, f'epoch_{epoch}', 'hr')
    sr_folder = os.path.join(args.results, f'epoch_{epoch}', 'sr')
    lr_upsample_folder = os.path.join(args.results, f'epoch_{epoch}', 'lr_upsample')
    os.makedirs(hr_folder, exist_ok=True)
    os.makedirs(sr_folder, exist_ok=True)
    os.makedirs(lr_upsample_folder, exist_ok=True)
    
    lr_folder = os.path.join(args.results, f'epoch_{epoch}', 'lr')
    os.makedirs(lr_folder, exist_ok=True)
    

    with torch.no_grad():
        for idx, (hr, lr, lr_upsample) in enumerate(valid_loader):
            hr = hr.to(args.device)  # 直接使用args.device字符串
            lr = lr.to(args.device)  # 直接使用args.device字符串
            lr_upsample = lr_upsample.to(args.device)  # 直接使用args.device字符串
            
            # lr_upsample = F.interpolate(lr, size=(args.high_size, args.high_size), mode='bicubic', align_corners=False)
            output = model(lr)

            for b in range(output.size(0)):
                 # ---------------------- 处理 HR 图像 ----------------------
                  # 处理 HR 图像
                 hr_tensor = hr[b].cpu().detach().permute(1, 2, 0)  # 转换为 (H, W, C) 格式
                 # 标准反归一化：假设输入范围是 [-1, 1]，映射到 [0, 255]
                 hr_tensor = (hr_tensor + 1) / 2  # 先转为 [0, 1]
                 hr_tensor = hr_tensor.clamp(0, 1)  # 确保数值在 [0, 1] 范围内（避免异常值）
                 hr_np = (hr_tensor * 255).numpy().astype(np.uint8)  # 先转为 numpy 数组，再转换类型
                 hr_path = os.path.join(hr_folder, f'{idx*output.size(0)+b}.png')
                 Image.fromarray(hr_np).save(hr_path)  # 保存 HR 图像
                #  hr_np = hr[b].cpu().detach().permute(1, 2, 0).numpy()  # 转换为 (H, W, C) 格式
                #  hr_np = (hr_np + 1) * 127.5  # 反归一化（假设输入归一化到 [-1, 1]）
                #  hr_np = hr_np.astype(np.uint8)
                #  hr_path = os.path.join(hr_folder, f'{idx*output.size(0)+b}_hr.png')
                #  Image.fromarray(hr_np).save(hr_path)  # 保存 HR 图像
                 # ---------------------- 处理 SR 图像（模型输出）----------------------
                 output_tensor = output[b].cpu().detach()  # 保留张量类型以便规范计算
                 output_tensor = output_tensor.permute(1, 2, 0)  # 转为 (H, W, C) 格式
                 # 标准反归一化：假设输入范围是 [-1, 1]，映射到 [0, 255]
                 output_tensor = (output_tensor + 1) / 2  # 先转为 [0, 1]
                 output_tensor = output_tensor.clamp(0, 1)  # 确保数值在 [0, 1] 范围内（避免异常值）
                 output_np = (output_tensor * 255).numpy().astype(np.uint8)  # 转为 [0, 255] 并转换类型
                 # 保存图像
                 sr_path = os.path.join(sr_folder, f'{idx*output.size(0)+b}.png')
                 Image.fromarray(output_np).save(sr_path)
                 
                 # ---------------------- 处理 LR 上采样图像 ----------------------
                 lr_upsample_tensor = lr_upsample[b].cpu().detach()  # 保留张量类型以便规范计算
                 lr_upsample_tensor = lr_upsample_tensor.permute(1, 2, 0)  # 转为 (H, W, C) 格式
                 # 标准反归一化：假设输入范围是 [-1, 1]，映射到 [0, 255]
                 lr_upsample_tensor = (lr_upsample_tensor + 1) / 2  # 先转为 [0, 1]
                 lr_upsample_tensor = lr_upsample_tensor.clamp(0, 1)  # 确保数值在 [0, 1] 范围内（避免异常值）
                 lr_upsample_np = (lr_upsample_tensor * 255).numpy().astype(np.uint8)  # 转为 [0, 255] 并转换类型
                 # 保存图像
                 lr_upsample_path = os.path.join(lr_upsample_folder, f'{idx*output.size(0)+b}.png')
                 Image.fromarray(lr_upsample_np).save(lr_upsample_path)
                #  # 不用处理  
                #  lr_upsample_np = lr_upsample[b].cpu().detach().permute(1, 2, 0).numpy()
                #  lr_upsample_np = (lr_upsample_np) * 255  # 反归一化
                #  lr_upsample_np = lr_upsample_np.astype(np.uint8)
                #  lr_upsample_path = os.path.join(lr_upsample_folder, f'{idx*output.size(0)+b}_lr_upsample.png')
                #  Image.fromarray(lr_upsample_np).save(lr_upsample_path)  # 保存 LR 上采样图像
                 # ---------------------- **新增：保存原始 LR 图像** ----------------------
                 # 处理 LR 图像
                 lr_tensor = lr[b].cpu().detach().permute(1, 2, 0)  # 转换为 (H, W, C) 格式
                 # 标准反归一化：假设输入范围是 [-1, 1]，映射到 [0, 255]
                 lr_tensor = (lr_tensor + 1) / 2  # 先转为 [0, 1]
                 lr_tensor = lr_tensor.clamp(0, 1)  # 确保数值在 [0, 1] 范围内（避免异常值）
                 lr_np = (lr_tensor * 255).numpy().astype(np.uint8)  # 先转为 numpy 数组，再转换类型
                 lr_path = os.path.join(lr_folder, f'{idx*output.size(0)+b}.png')
                 Image.fromarray(lr_np).save(lr_path)  # 保存原始 LR 图像
                 
                #  lr_np = lr[b].cpu().detach().permute(1, 2, 0).numpy()  # 提取当前样本的 LR 图像
                #  lr_np = (lr_np + 1) * 127.5  # 反归一化（与 HR/SR 保持一致）
                #  lr_np = lr_np.astype(np.uint8)
                #  lr_path = os.path.join(lr_folder, f'{idx*output.size(0)+b}_lr.png')
                #  Image.fromarray(lr_np).save(lr_path)  # 保存原始 LR 图像
                 
    fid_score = calculate_fid(hr_folder, sr_folder)
    lpips_score = calculate_lpips(hr_folder, sr_folder)
    psnr_score = calculate_psnr(hr_folder, sr_folder)
    ssim_score = calculate_ssim(hr_folder, sr_folder)
    print(f"valid-----Average PSNR 50张): {psnr_score:.2f}")
    print(f"valid-----Average SSIM 50张): {ssim_score:.4f}")
    print(f"valid-----FID 50张: {fid_score:.4f}")
    print(f"valid-----Lpips 50张: {lpips_score:.4f}")

    result_file = os.path.join(args.results, f'epoch_{epoch}', 'metrics.txt')
    with open(result_file, "w") as f:
        f.write(f"Average PSNR: {psnr_score:.2f}\n")
        f.write(f"Average SSIM: {ssim_score:.4f}\n")
        f.write(f"FID value: {fid_score:.4f}\n")
        f.write(f"Lpips value: {lpips_score:.4f}\n")
        
    save_model(model, epoch,args)
    
    return psnr_score, ssim_score, fid_score

# 训练函数（使用args.device）
def train(model, train_loader, optimizer, epoch, criterion, args):
    model.train()
    total_losses = []
    l1_losses = []
    hist_losses = []
    content_losses = []


    for batch_idx, (hr, lr, lr_image_unsampling) in enumerate(train_loader):
        hr = hr.to(args.device)  # 直接使用args.device字符串
        lr = lr.to(args.device)  # 直接使用args.device字符串
        lr_image_unsampling = lr_image_unsampling.to(args.device)  # 直接使用args.device字符串
        
        optimizer.zero_grad()
        output = model(lr)
        l1_loss, hist_loss, content_loss = criterion(hr, output)
        
        total_loss = l1_loss + hist_loss + content_loss
        total_loss.backward()
        optimizer.step()
        
        total_losses.append(total_loss.item())
        l1_losses.append(l1_loss.item())
        hist_losses.append(hist_loss.item())
        content_losses.append(content_loss.item())
        
		# # ---------------------- 新增：保存LR图像 ----------------------

    avg_total = sum(total_losses)/len(total_losses) if total_losses else 0.0
    avg_l1 = sum(l1_losses)/len(l1_losses) if l1_losses else 0.0
    avg_hist = sum(hist_losses)/len(hist_losses) if hist_losses else 0.0
    avg_content = sum(content_losses)/len(content_losses) if content_losses else 0.0
    
    print(f'Train Epoch: {epoch}\tTotal Loss: {avg_total:.6f} | L1: {avg_l1:.6f} | Hist: {avg_hist:.6f} | Content: {avg_content:.6f}')
    return avg_total, avg_l1, avg_hist, avg_content

# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, default='/newHome/S2_HHH/zzzz_Pixel_predictor/data/FFHQ/train')
    parser.add_argument('--test_dir', type=str, default='/newHome/S2_HHH/zzzz_Pixel_predictor/data/FFHQ/test')
    parser.add_argument('--valid_dir', type=str, default='/newHome/S2_HHH/zzzz_Pixel_predictor/data/FFHQ/valid')
    parser.add_argument('--low_size', type=int, default=32)
    parser.add_argument('--high_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0')  # 保留字符串类型参数
    parser.add_argument('--batch_size', type=int, default=75)
    parser.add_argument('--test_batch_size', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=36)
    parser.add_argument('--model_save', type=str, default='./pre_model/')
    parser.add_argument('--results', type=str, default='./results/')
    parser.add_argument('--valid_freq', type=int, default=2)
    args = parser.parse_args()

    # 注意：此处不转换为torch.device，直接使用字符串传递
    dtype = torch.cuda.FloatTensor if args.device.startswith('cuda') else torch.FloatTensor
    vgg16 = Vgg16().type(dtype).to(args.device)  # 直接使用args.device字符串
    
    
    dataset = CustomDataset(args, args.hr_dir)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = PixelPredictor(upscale_factor=int(args.high_size // args.low_size)).to(args.device)  # 直接使用args.device字符串
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    criterion = lambda x, y: image_compare_loss(x, y, device=args.device, vgg16=vgg16)  # 传递args.device字符串
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    args.results = os.path.join(args.results, f'FFHQ_{args.low_size}_{args.high_size}_model_{timestamp}')
    os.makedirs(args.results, exist_ok=True)
    
    args.model_save = os.path.join(args.model_save, f'FFHQ_{args.low_size}_{args.high_size}_model_{timestamp}')
    os.makedirs(args.model_save, exist_ok=True)
    
    loss_history = {'total_loss': [], 'l1_loss': [], 'hist_loss': [], 'content_loss': []}
    valid_history = {'psnr': [], 'ssim': [], 'fid': []}

    for epoch in range(1, args.epochs+1):
        # 传递args（包含device字符串）到train函数
        avg_total, avg_l1, avg_hist, avg_content = train(model, train_loader, optimizer, epoch, criterion, args)
        loss_history['total_loss'].append(avg_total)
        loss_history['l1_loss'].append(avg_l1)
        loss_history['hist_loss'].append(avg_hist)
        loss_history['content_loss'].append(avg_content)
        
        
        if epoch % args.valid_freq == 0:
            avg_psnr, avg_ssim, fid_value = valid(model, epoch, args)  # 仅传递args，内部使用args.device
            
            valid_history['psnr'].append(avg_psnr)
            valid_history['ssim'].append(avg_ssim)
            valid_history['fid'].append(fid_value)
            
            
    plt.figure(figsize=(10, 8))  # 适当增加高度以容纳多个子图
    # 总损失子图（第1行）
    plt.subplot(2, 2, 1)  # 2行2列，第1个位置
    plt.plot(loss_history['total_loss'], label='Total Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True)
    # L1损失子图（第2行）
    plt.subplot(2, 2, 2)  # 2行2列，第2个位置
    plt.plot(loss_history['l1_loss'], label='L1 Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('L1 Loss')
    plt.legend()
    plt.grid(True)
    
    # 直方图损失子图（第3行）
    plt.subplot(2, 2, 3)  # 2行2列，第3个位置
    plt.plot(loss_history['hist_loss'], label='Histogram Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Histogram Loss')
    plt.legend()
    plt.grid(True)
    # 内容损失子图（第4行）
    plt.subplot(2, 2, 4)  # 2行2列，第4个位置
    plt.plot(loss_history['content_loss'], label='Content Loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Content Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()  # 自动调整子图间距
    plt.savefig(os.path.join(args.results, 'training_loss_subplots.png'))
    plt.close()
    
    
    
    # 指标图
    plt.figure(figsize=(10, 15))
    # PSNR 子图
    plt.subplot(3, 1, 1)
    plt.plot(valid_history['psnr'], label='PSNR', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR Value')
    plt.title('Validation PSNR')
    plt.legend()
    plt.grid(True)
    # SSIM 子图
    plt.subplot(3, 1, 2)
    plt.plot(valid_history['ssim'], label='SSIM', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM Value')
    plt.title('Validation SSIM')
    plt.legend()
    plt.grid(True)
    # FID 子图
    plt.subplot(3, 1, 3)
    plt.plot(valid_history['fid'], label='FID', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('FID Value')
    plt.title('Validation FID')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results, 'validation_metrics.png'))
    plt.close()
    
    # **修改模型保存路径**
    results_folder_name = os.path.basename(args.results)  # 提取最后一级文件夹名称
    model_save_base = os.path.join(args.model_save, results_folder_name)  # 构建模型保存的基础路径
    os.makedirs(model_save_base, exist_ok=True)  # 创建模型保存的子文件夹
    model_save_filename = f'model_{timestamp}.pth'
    model_save_path = os.path.join(model_save_base, model_save_filename)  # 路径格式：pre_model/结果文件夹名称/model_时间戳.pth
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    print('-----------------Start-----------------')
    main()
	