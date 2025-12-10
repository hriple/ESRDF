from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode

from torchvision.transforms import functional as trans_fn

def resize_and_convert(img, size, resample):
    # print('img',img.size)
    if img.size[0] != size:
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img

class CustomDataset(Dataset):
    def __init__(self, args, dir,transform_hr=None, transform_lr=None):
        """
        Args:
            args: 包含参数的对象，需要有 args.high_size, args.low_size 属性
            transform_hr (callable, optional): 高分辨率图像预处理（如归一化）
            transform_lr (callable, optional): 低分辨率图像预处理（如下采样+归一化）
        """
        self.hr_paths = [os.path.join(dir, file) for file in os.listdir(dir) if file.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.high_size = args.high_size
        self.low_size = args.low_size
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        # 读取原始高分辨率图像（HR）
        hr_image = Image.open(self.hr_paths[idx]).convert("RGB")
        
		# 生成低分辨率图像（LR）
        if self.transform_lr is None:
            lr_transform = transforms.Compose([
                transforms.Resize(self.low_size, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            lr_image = lr_transform(hr_image)
        else:
            lr_image = self.transform_lr(hr_image)
        # ---------------------- 关键修改：[0, 1] ----------------------
        lr_image_4D = lr_image.unsqueeze(0)
        lr_upsample = F.interpolate(lr_image_4D, scale_factor=self.high_size/self.low_size, mode='bicubic', align_corners=False)
        lr_upsample = lr_upsample.squeeze(0)
        # ---------------------- 
        # # 使用反归一化后的张量生成 PIL 图像（用于 resize_and_convert 的 PIL 路径）
        # lr_t = (lr_image + 1)/2
        # lr_pil = transforms.ToPILImage()(lr_t)  # 转换为 PIL 图像（像素值 [0, 255]）
        # # 对 PIL 图像进行上采样（确保插值在整数像素值上进行）
        # lr_upsample_pil = resize_and_convert(lr_pil, self.high_size, InterpolationMode.BICUBIC)
        # # 将上采样后的 PIL 图像转换为张量并重新归一化
        # lr_upsample = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [0, 1] → [-1, 1]
        # ])(lr_upsample_pil)

        # 处理HR图像，调整到指定的高分辨率尺寸
        if self.transform_hr is None:
            hr_transform = transforms.Compose([
                transforms.Resize(self.high_size, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            hr_image = hr_transform(hr_image)
        else:
            hr_image = self.transform_hr(hr_image)

        

        return hr_image, lr_image, lr_upsample
