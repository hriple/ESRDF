import torch
import torch.nn as nn
import torch.nn.functional as F

# BSConv 层
# 用于初始特征提取，包含卷积、批量归一化和ReLU激活函数
class BSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BSConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)  # 卷积层
        self.bn = nn.BatchNorm2d(out_channels)  # 批量归一化层
        self.act = nn.ReLU()  # ReLU激活函数

    def forward(self, x):
        x = self.conv(x)  # 卷积操作
        x = self.bn(x)    # 批量归一化
        x = self.act(x)   # ReLU激活函数
        return x

# ARFU Block
# 残差特征提取块，包含两个卷积层和ReLU激活函数
class ARFU(nn.Module):
    def __init__(self, channels):
        super(ARFU, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 第一个卷积层
        self.relu = nn.ReLU()  # ReLU激活函数
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 第二个卷积层

    def forward(self, x):
        residual = x  # 保存输入作为残差
        x = self.conv1(x)  # 第一个卷积层
        x = self.relu(x)  # ReLU激活函数
        x = self.conv2(x)  # 第二个卷积层
        x = x + residual   # 残差连接
        return x

# ARFM Module
# 残差特征融合模块，包含三个ARFU块和一个卷积层
class ARFM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ARFM, self).__init__()
        self.arfu1 = ARFU(in_channels)  # 第一个ARFU块
        self.arfu2 = ARFU(in_channels)  # 第二个ARFU块
        self.arfu3 = ARFU(in_channels)  # 第三个ARFU块
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 融合卷积层
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 用于调整残差通道数

    def forward(self, x):
        residual = self.residual_conv(x)  # 调整残差通道数
        x = self.arfu1(x)  # 第一个ARFU块
        x = self.arfu2(x)  # 第二个ARFU块
        x = self.arfu3(x)  # 第三个ARFU块
        x = self.conv(x)   # 融合卷积层
        x = x + residual   # 残差连接
        return x

# PixelPredictor Network
# 超分辨率预测网络，包含特征提取、特征融合、上采样和最终输出
class PixelPredictor(nn.Module):
    def __init__(self, upscale_factor=8, in_channels=3, out_channels=3, base_channels=64):
        super(PixelPredictor, self).__init__()
        self.upscale_factor = upscale_factor  # 上采样倍数
        self.base_channels = base_channels  # 基础通道数

        # Initial BSConv
        # 初始特征提取
        self.bsconv = BSConv(in_channels, base_channels, kernel_size=3, padding=1)

        # ARFU Blocks
        # 残差特征提取块
        self.arfu1 = ARFU(base_channels)
        self.arfu2 = ARFU(base_channels)
        self.arfu3 = ARFU(base_channels)

        # Concat Layer (not explicitly needed, just for clarity)
        # 拼接层，用于将多个ARFU块的输出拼接在一起
        self.concat = nn.Identity()

        # ARFM Module
        # 残差特征融合模块
        self.arfm = ARFM(base_channels * 3, base_channels)  # 输入通道数为 base_channels * 3

        # Pixel Shuffle Layer
        # 像素重排层，用于上采样
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)

        # Bicubic Upsampling
        # 双三次插值上采样
        self.bicubic_upsample = lambda x: F.interpolate(x, scale_factor=upscale_factor, mode='bicubic', align_corners=False)

        # Final Convolution
        # 最终卷积层，用于调整通道数
        self.final_conv = nn.Conv2d(base_channels, out_channels * upscale_factor * upscale_factor, kernel_size=3, padding=1)

        # Convolution to match bicubic_out channels to out_channels
        # 卷积层，用于将双三次插值上采样的通道数调整为输出通道数
        self.bicubic_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Initial BSConv
        # 初始特征提取
        bsconv_out = self.bsconv(x)

        # ARFU Blocks
        # 残差特征提取
        arfu1_out = self.arfu1(bsconv_out)
        arfu2_out = self.arfu2(arfu1_out)
        arfu3_out = self.arfu3(arfu2_out)

        # Concatenate ARFU outputs
        # 拼接ARFU块的输出
        concat_out = torch.cat([arfu1_out, arfu2_out, arfu3_out], dim=1)
        concat_out = self.concat(concat_out)

        # ARFM Module
        # 残差特征融合
        arfm_out = self.arfm(concat_out)

        # Pixel Shuffle
        # 像素重排上采样
        pixel_shuffle_out = self.final_conv(arfm_out)
        pixel_shuffle_out = self.pixel_shuffle(pixel_shuffle_out)

        # Bicubic Upsampling
        # 双三次插值上采样
        bicubic_out = self.bicubic_upsample(bsconv_out)
        bicubic_out = self.bicubic_conv(bicubic_out)  # 调整双三次插值上采样的通道数

        # Add Bicubic and Pixel Shuffle outputs
        # 将双三次插值和像素重排的结果相加
        sr_out = pixel_shuffle_out + bicubic_out

        return sr_out