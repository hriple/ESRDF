import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as trans_fn
from torchvision.transforms import functional as trans_fu
from torchvision.transforms import InterpolationMode
from PIL import Image

# CBAM 结合了通道注意力和空间注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# BSConv 层
# 用于初始特征提取，包含卷积、批量归一化和ReLU激活函数
class BSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BSConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)  # 卷积层
        self.bn = nn.BatchNorm2d(out_channels)  # 批量归一化层
        self.act = nn.LeakyReLU(negative_slope=0.1)  # ReLU激活函数

    def forward(self, x):
        x = self.conv(x)  # 卷积操作
        x = self.bn(x)    # 批量归一化
        x = self.act(x)   # ReLU激活函数
        return x

# ARFU Block with cbam
class ARFUWithCBAM(nn.Module):
    def __init__(self, channels):
        super(ARFUWithCBAM, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 第一个卷积层
        # self.relu1 = nn.ReLU()  # ReLU激活函数
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)  # 替换为 LeakyReLU
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 第二个卷积层
        # self.relu2 = nn.ReLU()  # 残差后的激活
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)  # 替换为 LeakyReLU
        self.cbam = CBAM(channels, reduction_ratio=16, kernel_size=7)  # 使用CBAM

    def forward(self, x):
        residual = x  # 保存输入作为残差
        x = self.conv1(x)  # 第一个卷积层
        x = self.relu1(x)  # ReLU激活函数
        x = self.conv2(x)  # 第二个卷积层
        x = self.cbam(x)  # 应用CBAM注意力
        x = self.relu2(x + residual)
        return x

# ARFU Block
# 残差特征提取块，包含两个卷积层和LeakyReLU激活函数
class ARFU(nn.Module):
    def __init__(self, channels):
        super(ARFU, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 第一个卷积层
        # self.relu1 = nn.ReLU()  # ReLU激活函数
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)  # 替换为 LeakyReLU
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 第二个卷积层
        # self.relu2 = nn.ReLU()  # 残差后的激活
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)  # 替换为 LeakyReLU

    def forward(self, x):
        residual = x  # 保存输入作为残差
        x = self.conv1(x)  # 第一个卷积层
        x = self.relu1(x)  # ReLU激活函数
        x = self.conv2(x)  # 第二个卷积层
        x = x + residual   # 残差连接
        x = self.relu2(x)
        return x

# ARFM Module
# 残差特征融合模块，包含三个ARFU块和一个卷积层
class ARFM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ARFM, self).__init__()
        self.arfu1 = ARFUWithCBAM(in_channels)  # 第一个ARFU块
        self.arfu2 = ARFUWithCBAM(in_channels)  # 第二个ARFU块
        self.arfu3 = ARFUWithCBAM(in_channels)  # 第三个ARFU块
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 融合卷积层
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 用于调整残差通道数
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)  # 替换为 LeakyReLU

    def forward(self, x):
        residual = self.residual_conv(x)  # 调整残差通道数
        x = self.arfu1(x)  # 第一个ARFU块
        x = self.arfu2(x)  # 第二个ARFU块
        x = self.arfu3(x)  # 第三个ARFU块
        x = self.conv(x)   # 融合卷积层
        x = x + residual   # 残差连接
        x = self.relu1(x)
        return x

# PixelPredictor Network
# 超分辨率预测网络，包含特征提取、特征融合、上采样和最终输出
class PixelPredictor(nn.Module):
    def __init__(self, upscale_factor=4, in_channels=3, out_channels=3, base_channels=64):
        super(PixelPredictor, self).__init__()
        self.upscale_factor = upscale_factor  # 上采样倍数
        self.base_channels = base_channels  # 基础通道数

        # Initial BSConv
        # 初始特征提取
        self.bsconv = BSConv(in_channels, base_channels, kernel_size=3, padding=1)

        # ARFU Blocks
        # 残差特征提取块
        self.arfu1 = ARFUWithCBAM(base_channels)
        self.arfu2 = ARFUWithCBAM(base_channels)
        self.arfu3 = ARFUWithCBAM(base_channels)

        self.concat = nn.Identity()

        self.arfm = ARFM(base_channels * 3, base_channels)  # 输入通道数为 base_channels * 3

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

        bicubic_out = self.bicubic_upsample(x)
        # bicubic_out = self.bicubic_conv(bicubic_out)  # 调整双三次插值上采样的通道数

        # Add Bicubic and Pixel Shuffle outputs
        # 将双三次插值和像素重排的结果相加
        sr_out = pixel_shuffle_out + bicubic_out

        return sr_out