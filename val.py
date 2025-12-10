# 训练阶段包含了一个内部的验证循环，它会在每个验证频率 (val_freq) 后执行。在验证阶段，它会计算平均 PSNR 并保存验证图像。
# 在验证阶段保存了高分辨率（HR）、低分辨率（LR）、超分辨率（SR）和伪造（INF）图像。
import torch
import data as Data
import model as Model  # 模型定义模块
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_ddpm_40_160_DIV2K_test.json',
                        help='JSON file for configuration')  # 配置文件路径
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str)  # GPU ID
    parser.add_argument('-debug', '-d', action='store_true',default='')  # 调试模式
    parser.add_argument('-enable_wandb', action='store_true')  # 是否启用Wandb日志
    parser.add_argument('-log_wandb_ckpt', action='store_true')  # 是否在Wandb记录检查点
    parser.add_argument('-log_eval', action='store_true')  # 是否记录评估结果

    # 解析命令行参数
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # 设置CUDNN以提高计算性能
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # 设置日志记录器
    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # 初始化WandbLogger
    if opt['enable_wandb']:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # 加载数据集
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            # print(phase, dataset_opt)
            train_set = Data.create_dataset(dataset_opt, phase)
            # print(train_set)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)  # 创建训练集
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')
    

    # 创建模型
    diffusion = Model.create_model(opt)  # 根据配置创建模型
    logger.info('Initial Model Finished')  # 打印模型初始化完成信息

    # 训练过程
    current_step = diffusion.begin_step  # T
    current_epoch = diffusion.begin_epoch  # 迭代数
    n_iter = opt['train']['n_iter']  # 总迭代次数

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    # 设置噪声计划
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                # print('train_data',train_data)
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log,opt['train']['print_freq'] = 200
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log(opt)
                    # print("log", logs)
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)
                    if wandb_logger:
                        wandb_logger.log_metrics(logs)
                # 验证过程
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    avg_lpips = 0.0
                    
					# # 前面是 1e4,到1e5后面，就是 1e5进行一次验证
                    # if current_step == 3e5:
                    #      opt['train']['val_freq'] = 1e5

                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)
                    # 修改为创建子文件夹
                    subfolders = ['HR', 'LR', 'SR', 'INF', 'ADD']  # 添加需要的子文件夹类型
                    for folder in subfolders:
                         os.makedirs(os.path.join(result_path, folder), exist_ok=True)

                    # 设置验证阶段的噪声计划
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _, val_data in enumerate(val_loader):
                        # print('val_data',val_data)
                        p_avg_psnr = 0.0
                        p_avg_ssim = 0.0
                        p_avg_lpips = 0.0
                        
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=True)
                        # 获取当前视觉结果
                        visuals = diffusion.get_current_visuals()
                        for i in range(visuals['HR'].shape[0]):
                             HR_chunks = torch.chunk(visuals['HR'], chunks=visuals['HR'].shape[0], dim=0)
                             hr_img = Metrics.tensor2img(HR_chunks[i])
                             INF_chunks = torch.chunk(visuals['INF'], chunks=visuals['INF'].shape[0], dim=0)
                             inf_img = Metrics.tensor2img(INF_chunks[i])
                             LR_chunks = torch.chunk(visuals['LR'], chunks=visuals['LR'].shape[0], dim=0)
                             lr_img = Metrics.tensor2img(LR_chunks[i])
                            #  生成保存链接
                             hr_save_path = os.path.join(result_path, 'HR', f'{current_step}_{val_data["Index"][i]}.png')
                             inf_save_path = os.path.join(result_path, 'INF', f'{current_step}_{val_data["Index"][i]}.png')
                             lr_save_path = os.path.join(result_path, 'LR', f'{current_step}_{val_data["Index"][i]}.png')
                             sr_save_path = os.path.join(result_path, 'SR', f'{current_step}_{val_data["Index"][i]}.png')
                             add_save_path = os.path.join(result_path, 'ADD', f'{current_step}_{val_data["Index"][i]}.png')
                             
                             Metrics.save_img(hr_img, hr_save_path)
                             Metrics.save_img(inf_img, inf_save_path)
                             Metrics.save_img(lr_img, lr_save_path)
                             a = visuals['HR'].shape[0]
                             #  sr_img_indices = [-4, -3, -2, -1]
                             sr_img_indices = [i for i in range(-a, 0)]
                             sr_img_mode = 'grid'
                             if sr_img_mode == 'single':
                                  sr_img = visuals['SR']  # 超分辨率图像
                                  sample_num = sr_img.shape[0]  # 图像数量
                                  for iter in range(0, sample_num):
                                       Metrics.save_img(Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
                             else:
                                  sr_process_img = Metrics.tensor2img(visuals['SR'])  # uint8
                                #   Metrics.save_img(sr_process_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, val_data['Index'][i]))
                                  sr_img = Metrics.tensor2img(visuals['SR'][sr_img_indices[i]])
                                #   Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, val_data['Index'][i]))
                                
                                  Metrics.save_img(sr_img, sr_save_path)
                                  add_img, add_img_array = Metrics.add_images_and_normalize(visuals['SR'][sr_img_indices[i]], INF_chunks[i].squeeze(0))
                                  add_img.save(add_save_path)
                                #   add_img.save(os.path.join(result_path, f'{current_step}_{val_data["Index"][i]}_add_img.png'))
                                  
                             p_avg_psnr += Metrics.calculate_psnr(add_img_array, hr_img)
                             p_avg_ssim += Metrics.calculate_ssim(add_img_array, hr_img)
                             p_avg_lpips += Metrics.calculate_lpips(add_img_array, hr_img)
                             
                        p_avg_psnr = p_avg_psnr / float(visuals['HR'].shape[0])
                        p_avg_ssim = p_avg_ssim / float(visuals['HR'].shape[0])
                        p_avg_lpips = p_avg_lpips / float(visuals['HR'].shape[0])


                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (inf_img, sr_img, hr_img, add_img_array), axis=1), [2, 0, 1]), idx)
                        
                        # 计算PSNR,LR + Res图片跟 HR计算PSRN
                        avg_psnr += p_avg_psnr
                        avg_ssim += p_avg_ssim
                        avg_lpips += p_avg_lpips

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((inf_img, sr_img, hr_img, add_img_array), axis=1)
                            )
                            
                    # 计算PSNR平均值
                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    avg_lpips = avg_lpips / idx
                    hr_folder = os.path.join(result_path, 'HR')
                    add_folder = os.path.join(result_path, 'ADD')
                    fid = Metrics.calculate_fid(hr_folder, add_folder)

                    # 恢复训练阶段的噪声计划
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')

                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
                    logger.info('# Validation # Lpips: {:.4e}'.format(avg_lpips))
                    logger.info('# Validation # Fid: {:.4e}'.format(fid))
                    
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} ssim:{:.4e} lpips:{:.4e} fid:{:.4e}'.format(
                        current_epoch, current_step, avg_psnr, avg_ssim, avg_lpips, fid))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)
                    tb_logger.add_scalar('lpips', avg_lpips, current_step)
                    tb_logger.add_scalar('fid', fid, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_ssim': avg_ssim,
                            'validation/val_lpips': avg_lpips,
                            'validation/val_fid': fid,
                            'validation/val_step': val_step
                        })
                        val_step += 1
                # 保存模型和训练状态
                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch - 1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'], 'val_results')
        os.makedirs(result_path, exist_ok=True)
        # 设置验证阶段的噪声计划
        diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
        
        # 创建子文件夹（与训练阶段一致）
        subfolders = ['HR', 'LR', 'SR', 'INF', 'ADD']
        for folder in subfolders:
             os.makedirs(os.path.join(result_path, folder), exist_ok=True)
             
        for _, val_data in enumerate(val_loader):
            idx += 1
            print('idx',idx)
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()
            p_avg_psnr = 0.0
            p_avg_ssim = 0.0
            p_avg_lpips = 0.0
            for i in range(visuals['HR'].shape[0]):
                 
                 
                 HR_chunks = torch.chunk(visuals['HR'], chunks=visuals['HR'].shape[0], dim=0)
                 hr_img = Metrics.tensor2img(HR_chunks[i])
                 INF_chunks = torch.chunk(visuals['INF'], chunks=visuals['INF'].shape[0], dim=0)
                 inf_img = Metrics.tensor2img(INF_chunks[i])
                 LR_chunks = torch.chunk(visuals['LR'], chunks=visuals['LR'].shape[0], dim=0)
                 lr_img = Metrics.tensor2img(LR_chunks[i])
                 
                 #  生成保存链接
                 hr_save_path = os.path.join(result_path, 'HR', f'{current_step}_{val_data["Index"][i]}.png')
                 inf_save_path = os.path.join(result_path, 'INF', f'{current_step}_{val_data["Index"][i]}.png')
                 lr_save_path = os.path.join(result_path, 'LR', f'{current_step}_{val_data["Index"][i]}.png')
                 sr_save_path = os.path.join(result_path, 'SR', f'{current_step}_{val_data["Index"][i]}.png')
                 add_save_path = os.path.join(result_path, 'ADD', f'{current_step}_{val_data["Index"][i]}.png')
                #  保存图片
                 Metrics.save_img(hr_img, hr_save_path)
                 Metrics.save_img(inf_img, inf_save_path)
                 Metrics.save_img(lr_img, lr_save_path)
                 
                #  Metrics.save_img(inf_img, '{}/{}_{}_inf.png'.format(result_path, current_step, val_data['Index'][i]))
                #  Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, val_data['Index'][i]))
                #  Metrics.save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, val_data['Index'][i]))
                 
                 a = visuals['HR'].shape[0]
                 sr_img_indices = [i for i in range(-a, 0)]
                 
                 sr_img_mode = 'grid'
                 if sr_img_mode == 'single':
                       sr_img = visuals['SR']
                       sample_num = sr_img.shape[0]
                       for iter in range(0, sample_num):
                            Metrics.save_img(
				            Metrics.tensor2img(sr_img[iter]),
				            '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
                 else:
                      sr_process_img = Metrics.tensor2img(visuals['SR'])  # uint8
                    #   Metrics.save_img(sr_process_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, val_data['Index'][i]))
                      sr_img = Metrics.tensor2img(visuals['SR'][sr_img_indices[i]])
                      Metrics.save_img(sr_img, sr_save_path)
                    #   Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, val_data['Index'][i]))
                 add_img, add_img_array = Metrics.add_images_and_normalize(visuals['SR'][sr_img_indices[i]], INF_chunks[i].squeeze(0))
                 add_img.save(add_save_path)
                #  add_img.save(os.path.join(result_path, f'{current_step}_{val_data["Index"][i]}_add_img.png'))
                 p_avg_psnr += Metrics.calculate_psnr(add_img_array, hr_img)
                 p_avg_ssim += Metrics.calculate_ssim(add_img_array, hr_img)
                 p_avg_lpips += Metrics.calculate_lpips(add_img_array, hr_img)
                 
            p_avg_psnr = p_avg_psnr / float(visuals['HR'].shape[0])
            p_avg_ssim = p_avg_ssim / float(visuals['HR'].shape[0])
            p_avg_lpips = p_avg_lpips / float(visuals['HR'].shape[0])
            
            avg_psnr += p_avg_psnr
            avg_ssim += p_avg_ssim
            avg_lpips += p_avg_lpips

            if wandb_logger and opt['log_eval']:
                # wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr,eval_ssim)
                wandb_logger.log_eval_data(inf_img, add_img_array, hr_img, p_avg_psnr, p_avg_ssim, p_avg_lpips)
        # 计算平均PSNR和SSIM
        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        avg_lpips = avg_lpips / idx
        hr_folder = os.path.join(result_path, 'HR')
        add_folder = os.path.join(result_path, 'ADD')
        fid = Metrics.calculate_fid(hr_folder, add_folder)

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger.info('# Validation # Lpips: {:.4e}'.format(avg_lpips))
        logger.info('# Validation # Fid: {:.4e}'.format(fid))
        
        # print('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        # print('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        # print('# Validation # Lpips: {:.4e}'.format(avg_lpips))
        # print('# Validation # Fid: {:.4e}'.format(fid))
        
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr：{:.4e}, ssim：{:.4e}, lpips：{:.4e}, fid：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim, avg_lpips, fid))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim),
                'Lpips': float(avg_lpips),
                'Fid': float(fid)
            })
