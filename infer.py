# 推理采样
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
print(os.getcwd())
import sys
sys.path.append('/newHome/S2_HHH/gitte/ddpm_our/experiments')

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/newHome/S2_HHH/gitte/ddpm_our/config/test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')
    # Wandb（Weights & Biases）是一个机器学习实验跟踪和可视化工具，它可以帮助研究人员和开发人员跟踪模型训练过程中的各种指标、参数和输出。
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    # # 解析命令行参数
    args = parser.parse_args()
    opt = Logger.parse(args)
    # 将配置转换为NoneDict，以便在缺少键时返回None
    opt = Logger.dict_to_nonedict(opt)

    # 设置日志记录
    torch.backends.cudnn.enabled = True  # 启用CUDNN加速
    torch.backends.cudnn.benchmark = True  # 启用CUDNN基准测试

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)  # 设置训练日志
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)  # 设置验证日志
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # 初始化WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # 加载数据集
    for phase, dataset_opt in opt['datasets'].items():
        # opt['datasets'].items() 的环境
        # phase,dataset_opt train {'name': 'FFHQ', 'mode': 'HR', 'dataroot': 'dataset/ffhq_64_512', 'datatype': 'img', 'l_resolution': 64, 'r_resolution': 512, 'batch_size': 2, 'num_workers': 8, 'use_shuffle': True, 'data_len': -1}
        # phase,dataset_opt val {'name': 'CelebaHQ', 'mode': 'LRHR', 'dataroot': 'dataset/celebahq_64_512', 'datatype': 'img', 'l_resolution': 64, 'r_resolution': 512, 'data_len': 50}
        # print('phase,dataset_opt', phase, dataset_opt)

        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # print('opt', opt)
    # 初始化模型
    diffusion = Model.create_model(opt)  # 根据配置创建模型
    logger.info('Initial Model Finished')

    # 设置模型的噪声计划
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    # 创建结果保存路径
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _, val_data in enumerate(val_loader):

        idx += 1
        diffusion.feed_data(val_data)  # 将数据喂给模型
        diffusion.test(continous=True)  # 进行连续推理
        visuals = diffusion.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # 将HR图像从张量转换为图像
        fake_img = Metrics.tensor2img(visuals['INF'])  # 将INF图像从张量转换为图像

        sr_img_mode = 'grid'  # 设置SR图像模式
        if sr_img_mode == 'single':
            # 单张图像系列
            sr_img = visuals['SR']  # 获取SR图像
            sample_num = sr_img.shape[0]  # 获取样本数量
            # 遍历每个样本
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # 网格图像
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            # 获取SR图像
            Metrics.save_img(
                sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
            # 保存处理中的SR图像
            Metrics.save_img(
                Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

        # 保存HR图像
        Metrics.save_img(
            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
        # 保存伪造图像
        Metrics.save_img(
            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

        # 如果启用Wandb，记录评估数据
        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img)
    print('idx', idx)
    # 如果启用Wandb，记录评估表格
    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
