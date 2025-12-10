import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def _cosine_to_quadratic_beta(cosine_start, cosine_end, quadratic_end, n_timestep, cosine_frac):
    # 初始化噪声方差数组
    betas = np.zeros(n_timestep, dtype=np.float64)
    
    # 计算余弦阶段的时间步数
    cosine_time = int(n_timestep * cosine_frac)
    
    # 余弦阶段：从 cosine_start 线性增加到 cosine_end
    betas[:cosine_time] = np.linspace(cosine_start, cosine_end, cosine_time, dtype=np.float64)
    
    # 二次阶段：从 cosine_end 二次增加到 quadratic_end
    quadratic_time = n_timestep - cosine_time
    quadratic_steps = np.arange(1, quadratic_time + 1)
    betas[cosine_time:] = cosine_end + (quadratic_steps / quadratic_time) ** 2 * (quadratic_end - cosine_end)
    
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    # 二次调度的曲线比线性调度更平滑，有助于减少采样过程中的震荡。相比线性调度，二次调度的实现稍复杂，需要平方和开方操作。在某些任务中，二次调度可能导致噪声变化过快或过慢，影响生成质量。
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    # 线性调度是最直观和最容易实现的调度方式。线性调度的噪声变化速度固定，可能无法适应复杂的扩散过程。
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    # 预热调度,在前 warmup_ratio 的时间步中，噪声缓慢增加。在剩余的时间步中，噪声快速增加。预热阶段允许模型在早期逐步适应噪声，避免过早引入大量噪声导致的不稳定。
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    # 所有时间步的噪声方差 β_t均为常数 linear_end。实现非常简单，计算开销小。噪声方差固定，无法根据时间步动态调整。
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
        
	# 倒数序列的噪声变化较为平滑，有助于稳定采样过程。在某些任务中，倒数序列可能导致噪声变化过慢，影响采样效率。
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    # 噪声方差 β_t通过余弦函数生成，曲线平滑且在两端逐渐趋于平稳。余弦调度的噪声变化非常平滑，有助于减少采样过程中的震荡。需要计算余弦函数和归一化操作，实现稍复杂。
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    elif schedule == "linear_cosine":
        betas1 = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)

        steps = n_timestep + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + cosine_s) / (1 + cosine_s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas2 = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas2 = np.clip(betas2, a_min=0, a_max=0.999)

        betas = np.add(betas1, np.add(betas2, betas2))
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule == "cosine_to_quadratic":
        cosine_frac = 0.1  # 余弦阶段占总时间步的比例
        cosine_end = 0.02  # 余弦阶段结束时的噪声方差
        quadratic_end = 0.999  # 二次阶段结束时的噪声方差

        # 余弦阶段
        cosine_time = int(n_timestep * cosine_frac)
        timesteps = torch.arange(cosine_time + 1, dtype=torch.float64) / cosine_time
        alphas_cumprod = torch.cos(((timesteps + cosine_s) / (1 + cosine_s)) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas_cosine = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_cosine = betas_cosine.clamp(max=0.999).numpy()

        # 二次阶段
        quadratic_time = n_timestep - cosine_time
        quadratic_steps = np.arange(1, quadratic_time + 1)
        betas_quadratic = cosine_end + (quadratic_steps / quadratic_time) ** 2 * (quadratic_end - cosine_end)

        # 组合两个阶段
        betas = np.concatenate((betas_cosine, betas_quadratic))
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# 从一个数组中提取特定时间步 t 的值，并将其扩展到与输入图像 x_t相同的形状
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 生成一个随机噪声
def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()



import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

def save_images(x_start, noise, x_noisy, x_recon, save_dir, prefix=""):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 将张量归一化到 [0, 1] 范围
    def normalize(tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # 保存原始图像
    x_start_normalized = normalize(x_start)
    save_image(x_start_normalized, os.path.join(save_dir, f"{prefix}_x_start.png"))

    # 保存噪声图像
    noise_normalized = normalize(noise)
    save_image(noise_normalized, os.path.join(save_dir, f"{prefix}_noise.png"))

    # 保存加噪图像
    x_noisy_normalized = normalize(x_noisy)
    save_image(x_noisy_normalized, os.path.join(save_dir, f"{prefix}_x_noisy.png"))

    # 保存去噪后图像
    x_recon_normalized = normalize(x_recon)
    save_image(x_recon_normalized, os.path.join(save_dir, f"{prefix}_x_recon.png"))

    print(f"Images saved to {save_dir} with prefix '{prefix}'")


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.conditional = conditional
        self.loss_type = loss_type
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

	# 在这里，一般 q 表示前向过程。，p代表逆向过程。但是有一部分前向过程公式会是逆向过程中的一步。
    # 扩散过程是将原始图像 x_0,逐步转化为噪声图像 x_t的过程，这个过程可以通过一系列条件概率分布 q(x_t​∣x_t−1) 来描述。
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

	# 根据当前时间步 t 的噪声图像 x_t和对应的噪声 ϵ，预测出原始图像 x_0, 是前向过程 x_t 和 x_0 公式的 逆转。
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

	# 用于计算给定当前噪声图像 x_t 和原始图像 x_0 的情况下，下一步图像 x_t−1的后验分布的均值、方差和方差的对数。
    # 决定了如何从 x_t和 x_0​生成 x_t−1。μ(x_t,x_0)
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

	# p_mean_variance 方法计算了在给定时间步 t 和噪声图像 x_t的情况下，下一步图像 x_t−1的均值和方差。
    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        if condition_x is not None:
            # noise=self.denoise_fn(torch.cat([condition_x, x], dim=1) 使用训练好的 去噪模型，预测noise
            # 根据当前时间步 t 的噪声图像 x_t和对应的噪声 ϵ，预测出原始图像 x_0, 是前向过程 x_t 和 x_0 公式的 逆转。
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), t))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
            
		# model_mean：下一步图像 x_t−1的均值。
        # posterior_variance 下一步图像 x_t−1的方差。
        # posterior_log_variance 下一步图像 x_t−1的方差的对数。
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    # p_sample 方法实现了单步的去噪过程。它根据当前时间步 t 和噪声图像 x_t，计算出下一步的图像 x_t−1。
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None):
        b, *_, device = *x.shape, x.device
        # p_mean_variance 方法计算了在给定时间步 t 和噪声图像 x_t的情况下，下一步图像 x_t−1的均值和方差。
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    # p_sample_loop 方法实现了从时间步 T 到时间步 0 的逆过程，逐步去噪并生成图像
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))

        if not self.conditional:
            shape = x_in
            b = shape[0]
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                # p_sample 方法实现了单步的去噪过程。它根据当前时间步 t 和噪声图像 x_t，计算出下一步的图像 x_t−1。
                img = self.p_sample(img, torch.full(
                    (b,), i, device=device, dtype=torch.long))
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
            return img
        else:
            x = x_in
            shape = x.shape
            b = shape[0]
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, torch.full(
                    (b,), i, device=device, dtype=torch.long), condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

	# ----------下面是任务型 函数
	# 用于生成新的图像样本,该代码，分为 图像生成 和 图像超分辨率重建任务
    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)
	# 测试用的，开始 去噪重建图片
    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        # p_sample_loop 是扩散模型的核心逆过程实现，它逐步从噪声图像中恢复出原始图像。
        return self.p_sample_loop(x_in, continous)

	# 用于在两个给定的图像 x1和 x2之间进行插值，并生成一个新的图像。这个方法通过扩散模型的逆过程（即去噪过程）逐步从噪声图像中恢复出插值后的图像。
    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img
    
	
	# 前向加噪，q_sample 函数是扩散模型中的一个关键部分，用于实现前向扩散过程。它的作用是根据给定的原始图像 x0和 时间步 t，生成对应的噪声图像 xt。
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # fix gama
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )
        # random gama
        # x_shape = x_start.shape
        # l = self.alphas_cumprod .gather(-1, t)
        # r = self.alphas_cumprod .gather(-1, t+1)
        # gama = (r - l) * torch.rand(0, 1) + l
        # gama = gama.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))
        # return (
        #     nq.sqrt(gama) * x_start + nq.sqrt(1-gama)* noise
        # )
        
	# p_losses 函数是扩散模型中的一个关键部分，用于计算训练过程中的损失函数。
	# 它的作用是通过前向扩散过程生成噪声图像，然后利用去噪模型（denoise_fn）预测噪声，并计算预测噪声与真实噪声之间的损失。
    def p_losses(self, x_in, noise=None):
        # x_start = x_in['HR']
        x_start = x_in['HR'] - x_in['SR']
        
        [b, c, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        
        
		# 噪声，根据 x_start 形状
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 加噪图片
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
		# 开始去噪，输入 x_t 和 t
        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, t)
        else:
            x_recon = self.denoise_fn(torch.cat([x_in['SR'], x_noisy], dim=1), t)
        # x_recon 可以是噪声，也可以是x_0,看Loss的引导。
            
        # save_images(x_start, noise, x_noisy, x_recon, save_dir="/newHome/S2_HHH/gitte", prefix="example")
            
        loss = self.loss_func(noise, x_recon)

        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
