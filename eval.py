import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
import torch
from torchvision import transforms
from pytorch_fid import fid_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default=r'eval\results')
    parser.add_argument('--real_folder', type=str, default=r'D:\Download\Image-Super-Resolution-via-Iterative-Refinement-master\eval\real')
    parser.add_argument('--generated_folder', type=str, default=r'D:\Download\Image-Super-Resolution-via-Iterative-Refinement-master\eval\fake')
    args = parser.parse_args()

    # 使用 glob 模块获取所有高分辨率（hr）图像和超分辨率（sr）图像的路径列表。
    real_names = list(glob.glob('{}/*_hr.png'.format(args.path)))
    fake_names = list(glob.glob('{}/*_sr.png'.format(args.path)))
    # 对图像路径列表进行排序，确保成对的图像是对应的。
    real_names.sort()
    fake_names.sort()

    # 初始化PSNR和SSIM的平均值变量和计数器。
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    for rname, fname in zip(real_names, fake_names):
        idx += 1
        ridx = rname.rsplit("_hr")[0]
        fidx = fname.rsplit("_sr")[0]

        assert ridx == fidx, f'Image ridx:{ridx}!=fidx:{fidx}'
        try:
            hr_img = np.array(Image.open(rname))
            sr_img = np.array(Image.open(fname))
        except Exception as e:
            print(f"打开图像文件 {rname} 或 {fname} 时出错: {e}")
            continue
        # 计算每对图像的PSNR和SSIM值。
        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        ssim = Metrics.calculate_ssim(sr_img, hr_img)
        avg_psnr += psnr
        avg_ssim += ssim
        if idx % 20 == 0:
            print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}'.format(idx, psnr, ssim))

    assert idx != 0, "idx为0，请检查文件夹里的数据数量"

    # 计算所有图像的平均PSNR和SSIM值。
    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx

    # log
    print('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    print('# Validation # SSIM: {:.4e}'.format(avg_ssim))

    # 指定批次大小、计算设备和特征维度
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dims = 2048  # Inception网络的默认特征维度
    real_images_folder = args.real_folder
    generated_images_folder = args.generated_folder
    # 计算FID值
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_folder, generated_images_folder],
        batch_size=batch_size,
        device=device,
        dims=dims
    )
    print('# Validation # FID value:', fid_value)