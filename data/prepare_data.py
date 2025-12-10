import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb
import numpy as np
import time
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import torch

# 用于批量处理图像，将它们调整到不同的分辨率，并可选择将处理后的图像保存到文件系统或LMDB数据库中

# 调整图像大小并进行中心裁剪。
# 参数:
#     img (PIL.Image): 要处理的PIL图像对象。
#     size (int): 目标图像的宽度和高度（假设图像是正方形）。
#     resample (int): 重采样滤波器，用于调整图像大小时的插值方法。
#
# 返回:
#     PIL.Image: 调整大小并进行中心裁剪后的图像。
def resize_and_convert(img, size, resample):
    # 检查图像的宽度是否与目标大小不同
    if (img.size[0] != size):
        # 使用指定的重采样方法调整图像的大小
        img = trans_fn.resize(img, size, resample)
        # 对调整大小后的图像进行中心裁剪，确保图像是正方形
        img = trans_fn.center_crop(img, size)
    return img


# 将图像转换为字节数据。
# 将PIL图像对象转换为字节数据。
#
# 参数:
# img (PIL.Image): 要转换的PIL图像对象。
#
# 返回:
# bytes: 图像的字节数据。
def image_convert_bytes(img):
    # 创建一个BytesIO对象，用于在内存中存储图像数据
    buffer = BytesIO()
    # 将图像保存到BytesIO对象中，格式为PNG
    img.save(buffer, format='png')
    return buffer.getvalue()


# 将图像调整到多个不同的大小，并可选择将图像数据转换为字节数据。
#
#     参数:
#     img (PIL.Image): 要处理的PIL图像对象。
#     sizes (tuple): 一个包含两个整数的元组，指定低分辨率和高分辨率的大小。
#     resample (int): 重采样滤波器，用于调整图像大小时的插值方法。双三次插值法
#     lmdb_save (bool): 是否将图像数据转换为字节数据以便于存储到LMDB数据库。
#
#     返回:
#     list: 包含三个图像对象或字节数据的列表。
def resize_multiple(img, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False):
    # 调整图像到低分辨率大小
    lr_img = resize_and_convert(img, sizes[0], resample)
    # 调整图像到高分辨率大小
    hr_img = resize_and_convert(img, sizes[1], resample)
    # 将低分辨率图像进一步上采样到高分辨率大小
    sr_img = resize_and_convert(lr_img, sizes[1], resample)
    
    # 如果需要将图像数据保存到LMDB数据库，则将图像转换为字节数据
    if lmdb_save:
        lr_img = image_convert_bytes(lr_img)
        hr_img = image_convert_bytes(hr_img)
        sr_img = image_convert_bytes(sr_img)

    # 返回包含三个图像的列表，它们分别是低分辨率图像、高分辨率图像和上采样图像
    return [lr_img, hr_img, sr_img]


# 处理单个图像文件，将其打开、转换为RGB格式，并调整到多个不同的大小。
#
#     参数:
#     img_file (Path): 要处理的图像文件的路径。
#     sizes (tuple): 一个包含两个整数的元组，指定低分辨率和高分辨率的大小。
#     resample (int): 重采样滤波器，用于调整图像大小时的插值方法。
#     lmdb_save (bool): 是否将图像数据转换为字节数据以便于存储到LMDB数据库。
#
#     返回:
#     tuple: 包含图像文件名（不含扩展名）和处理后的图像列表的元组。
def resize_worker(img_file, sizes, resample, lmdb_save=False):
    img = Image.open(img_file)
    img = img.convert('RGB')
    # 调用resize_multiple函数，调整图像到指定的多个大小
    out = resize_multiple(img, sizes=sizes, resample=resample, lmdb_save=lmdb_save)

    return img_file.name.split('.')[0], out


# 用于管理和同步多进程图像处理任务的状态。
#
#     属性:
#     resize_fn (function): 用于调整图像大小的函数。
#     lmdb_save (bool): 指示是否将图像数据保存到LMDB数据库。
#     out_path (str): 处理后的图像保存的路径。
#     env (lmdb.Environment): LMDB环境对象，用于数据库操作。
#     sizes (tuple): 包含低分辨率和高分辨率大小的元组。
#     counter (multiprocessing.RawValue): 用于跟踪处理图像数量的共享整数。
#     counter_lock (multiprocessing.Lock): 用于同步访问共享计数器的锁。
class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, env, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.sizes = sizes

        # 创建一个共享整数用于跟踪处理的图像数量
        self.counter = RawValue('i', 0)
        # 创建一个锁用于同步访问共享计数器
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value


# 在多进程环境中处理图像文件子集。
#
#     参数:
#     wctx (WorkingContext): 包含共享状态和函数的上下文对象。
#     file_subset (list): 要处理的图像文件子集的列表。
#
#     功能:
#     - 遍历图像文件子集，对每个文件调用resize_fn进行大小调整。
#     - 根据配置，将处理后的图像保存到文件系统或LMDB数据库。
#     - 维护并更新处理的图像数量。
def prepare_process_worker(wctx, file_subset):
    for file in file_subset:
        i, imgs = wctx.resize_fn(file)
        # 解包处理后的图像：lr_img（低分辨率），hr_img（高分辨率），sr_img（超分辨率）
        lr_img, hr_img, sr_img = imgs
        # 如果配置为不保存到LMDB数据库，将图像保存到文件系统
        if not wctx.lmdb_save:
            lr_img.save(
                '{}/lr_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], i.zfill(5)))
            hr_img.save(
                '{}/hr_{}/{}.png'.format(wctx.out_path, wctx.sizes[1], i.zfill(5)))
            sr_img.save(
                '{}/sr_{}_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], wctx.sizes[1], i.zfill(5)))
        else:
            # 如果配置为保存到LMDB数据库，将图像转换为字节数据并保存
            with wctx.env.begin(write=True) as txn:
                txn.put('lr_{}_{}'.format(
                    wctx.sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                txn.put('hr_{}_{}'.format(
                    wctx.sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                txn.put('sr_{}_{}_{}'.format(
                    wctx.sizes[0], wctx.sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
        curr_total = wctx.inc_get()
        if wctx.lmdb_save:
            with wctx.env.begin(write=True) as txn:
                txn.put('length'.encode('utf-8'), str(curr_total).encode('utf-8'))


# 检查给定的线程列表中的所有线程是否都已停止运行。
#
#     参数:
#     worker_threads (list): 要检查的线程列表。
#
#     返回:
#     bool: 如果所有线程都已停止运行，则返回True；否则返回False。
def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True


# 用于准备图像处理流程，包括解析参数、创建输出目录、启动多进程等。
# 准备图像处理流程，包括创建目录、启动多进程处理和保存处理后的图像。
#
#     参数:
#     img_path (str): 原始图像所在的路径。
#     out_path (str): 处理后的图像保存的路径。
#     n_worker (int): 用于处理图像的工作进程数。
#     sizes (tuple): 包含低分辨率和高分辨率大小的元组。
#     resample (int): 重采样滤波器，用于调整图像大小时的插值方法。
#     lmdb_save (bool): 是否将图像数据保存到LMDB数据库。
#
#     功能:
#     - 创建必要的目录以保存处理后的图像。
#     - 根据工作进程数启动多进程处理。
#     - 将处理后的图像保存到文件系统或LMDB数据库。
def prepare(img_path, out_path, n_worker, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False):
    # 创建一个预处理函数，用于调整图像大小并根据需要转换为字节数据
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample, lmdb_save=lmdb_save)
    # 获取所有图像文件的路径
    files = [p for p in Path(
        '{}'.format(img_path)).glob(f'**/*')]

    # 如果不保存到LMDB数据库，创建必要的目录
    if not lmdb_save:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs('{}/lr_{}'.format(out_path, sizes[0]), exist_ok=True)
        os.makedirs('{}/hr_{}'.format(out_path, sizes[1]), exist_ok=True)
        os.makedirs('{}/sr_{}_{}'.format(out_path,
                                         sizes[0], sizes[1]), exist_ok=True)
    else:
        # 如果保存到LMDB数据库，打开LMDB环境
        env = lmdb.open(out_path, map_size=1024 ** 4, readahead=False)

    # 根据工作进程数启动多进程处理
    if n_worker > 1:
        # 准备数据子集
        multi_env = None
        if lmdb_save:
            multi_env = env

        file_subsets = np.array_split(files, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, sizes)

        # 启动工作进程并监控结果
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, file_subsets[i]))
            proc.start()
            worker_threads.append(proc)

        total_count = str(len(files))
        while not all_threads_inactive(worker_threads):
            print("\r{}/{} images processed".format(wctx.value(), total_count), end=" ")
            time.sleep(0.1)

    else:
        # 如果只有一个工作进程，使用单线程处理
        total = 0
        for file in tqdm(files):
            i, imgs = resize_fn(file)
            lr_img, hr_img, sr_img = imgs
            if not lmdb_save:
                lr_img.save(
                    '{}/lr_{}/{}.png'.format(out_path, sizes[0], i.zfill(5)))
                hr_img.save(
                    '{}/hr_{}/{}.png'.format(out_path, sizes[1], i.zfill(5)))
                sr_img.save(
                    '{}/sr_{}_{}/{}.png'.format(out_path, sizes[0], sizes[1], i.zfill(5)))
            else:
                with env.begin(write=True) as txn:
                    txn.put('lr_{}_{}'.format(
                        sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                    txn.put('hr_{}_{}'.format(
                        sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                    txn.put('sr_{}_{}_{}'.format(
                        sizes[0], sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
            total += 1
            if lmdb_save:
                with env.begin(write=True) as txn:
                    txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='/newHome/S2_HHH/data/test_1200')
    parser.add_argument('--out', '-o', type=str,
                        default='../dataset/test_1200')

    parser.add_argument('--size', type=str, default='32,128')
    parser.add_argument('--n_worker', type=int, default=3)
    parser.add_argument('--resample', type=str, default='bicubic')
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]

    args.out = '{}_{}_{}'.format(args.out, sizes[0], sizes[1])
    prepare(args.path, args.out, args.n_worker, sizes=sizes, resample=resample, lmdb_save=args.lmdb)
