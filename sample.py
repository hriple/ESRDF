# 两种模式：训练和评估（生成）。
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
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sample_sr3_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)  # GPU
    parser.add_argument('-debug', '-d', action='store_true')  # 调试模式
    parser.add_argument('-enable_wandb', action='store_true')  # 是否启用wandb
    parser.add_argument('-log_wandb_ckpt', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
        val_step = 0
    else:
        wandb_logger = None

    # 加载数据集
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            # 创建训练数据集
            train_set = Data.create_dataset(dataset_opt, phase)
            # 创建训练数据加载器
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # 创建模型
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # 训练
    current_step = diffusion.begin_step  # 当前迭代次数
    current_epoch = diffusion.begin_epoch  # 当前训练周期
    n_iter = opt['train']['n_iter']  # 总迭代次数
    sample_sum = opt['datasets']['val']['data_len']  # 验证集样本数量

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))  # 记录恢复训练的信息

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])  # 设置噪声计划

    if opt['phase'] == 'train':
        while current_step < n_iter:  # 当当前步数小于总迭代次数时，继续训练
            current_epoch += 1  # 每完成一个训练循环，增加一个训练周期
            for _, train_data in enumerate(train_loader):  # 遍历训练数据加载器
                current_step += 1
                if current_step > n_iter:
                    break
                # 将训练数据喂给模型
                diffusion.feed_data(train_data)
                # 优化模型参数
                diffusion.optimize_parameters()
                # log 每间隔一定的步数，记录日志
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # 每间隔一定的步数，进行一次验证
                if current_step % opt['train']['val_freq'] == 0:
                    # 构建验证结果的存储路径
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    # 设置验证阶段的噪声计划
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')

                    # 遍历验证集的所有样本
                    for idx in range(sample_sum):
                        # 生成样本
                        diffusion.sample(continous=False)
                        # 获取当前的视觉结果
                        visuals = diffusion.get_current_visuals(sample=True)
                        # 将结果张量转换为图像（uint8格式）
                        sample_img = Metrics.tensor2img(
                            visuals['SAM'])  # uint8
                        # 生成
                        Metrics.save_img(
                            sample_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(sample_img, [2, 0, 1]),
                            idx)

                        if wandb_logger:
                            wandb_logger.log_image(f'validation_{idx}', sample_img)

                    # 验证结束后，恢复训练阶段的噪声计划
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')

                # 每间隔一定的步数，保存一次模型
                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

        # 保存模型
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')

        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        # 初始化一个空列表，用于存储生成的图像
        sample_imgs = []
        for idx in range(sample_sum):
            idx += 1
            # 连续生成样本，用于评估
            diffusion.sample(continous=True)
            # 获取当前的视觉结果
            visuals = diffusion.get_current_visuals(sample=True)

            # 设置显示图像的模式
            show_img_mode = 'grid'
            if show_img_mode == 'single':
                # 单张图像序列
                sample_img = visuals['SAM']  # 获取视觉结果张量，uint8格式
                sample_num = sample_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        # 将张量转换为图像
                        Metrics.tensor2img(sample_img[iter]),
                        '{}/{}_{}_sample_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sample_img = Metrics.tensor2img(visuals['SAM'])  # uint8
                Metrics.save_img(
                    sample_img, '{}/{}_{}_sample_process.png'.format(result_path, current_step, idx))
                # 保存最终图像
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SAM'][-1]),
                    '{}/{}_{}_sample.png'.format(result_path, current_step, idx))
            # 将最终图像添加到列表中
            sample_imgs.append(Metrics.tensor2img(visuals['SAM'][-1]))

        if wandb_logger:
            wandb_logger.log_images('eval_images', sample_imgs)
