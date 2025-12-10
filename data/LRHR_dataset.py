from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util


class LRHRDataset(Dataset):
    # 定义LRHRDataset类的初始化方法__init__。
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        # self.need_LR = True
        self.split = split
        # self.data_len 接受的值就是 3，因此在调用处传输进来的是3
        # print('self.data_len_1',self.data_len)

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            # 到数据集文件夹里取文件
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            # 到数据集文件夹里取文件
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                # 到数据集文件夹里取文件
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            # 为什么要 len(self.hr_path) 这个做为 dataset_len
            self.dataset_len = len(self.hr_path)
            # print('train_set',self.data_len)
            # -1拿所有数据集
            if self.data_len <= 0:
                self.data_len = self.dataset_len
                # print('train_set',self.data_len)
            else:
                self.data_len = min(self.data_len, self.dataset_len)
                # print('val_set',self.data_len)
                # print('self.data_len',self.data_len)
                # print('self.dataset_len',self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))
    # 定义__len__方法，返回数据集的长度，即data_len。 在这个LRHR_dataset里面，已经固定了返回 data_len 张图像了。因此shuffle也至少在那几张里随机
    def __len__(self):
        return self.data_len
    # 定义__getitem__方法，这是Dataset类的核心方法，用于按索引获取数据。
    def __getitem__(self, index):
        img_HR = None
        img_LR = None
        # print('self.need_LR',self.need_LR)
        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            # 三通道打开RGB,图像的像素值是按照 0-255 的整数范围来表示的。
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        # need_LR指定是否需要加载低分辨率图像。
        if self.need_LR:
            # transform_augment 函数的主要作用是对输入的图像列表进行预处理和数据增强。
			# 在训练阶段，它会对图像进行水平翻转以增加数据集的多样性。
			# 此外，它还根据 min_max 参数对图像的像素值进行缩放和偏移，以适应不同的数值范围。
            [img_LR, img_SR, img_HR] = Util.transform_augment([img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment([img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
