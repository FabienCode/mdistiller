import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
from mdistiller.engine.mvkd_utils import Model
from tqdm import tqdm

class MVKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(MVKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.MVKD.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.MVKD.LOSS.FEAT_WEIGHT
        self.hint_layer = cfg.MVKD.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )

        # feature restoration
        # Diffusion config
        self.infer_weight = cfg.MVKD.LOSS.INFER_WEIGHT
        self.rec_weight = cfg.MVKD.LOSS.REC_WEIGHT
        timesteps = 1000
        sampling_timesteps = cfg.MVKD.NUM_TIMESTEPS
        self.betas = cosine_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.timesteps, = self.betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.d_scale = cfg.MVKD.D_SCALE
        self.f_t_shapes = feat_t_shapes
        t_b, t_c, t_w, t_h = feat_t_shapes[self.hint_layer]
        self.rec_module = Model(ch=t_c, out_ch=t_c, num_res_blocks=2, attn_resolutions=[4], in_channels=t_c,
                                resolution=t_w, dropout=0.0)
        self.make_schedule(self.sampling_timesteps, ddim_discretize="uniform", ddim_eta=self.ddim_sampling_eta, verbose=True)
        # self.prepare_noise_feature

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters()) + list(self.rec_module.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        for p in self.rec_module.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        cur_epoch = kwargs['epoch']
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        f_t = feature_teacher["feats"][self.hint_layer]

        if cur_epoch > 200:
            # 利用训练好的diffusion模型从随机噪声中生成不同的feature.
            f_new, f_inter = self.ddim_sampling(f_t, f_t) #
            # f_new = self.ddim_sample(f_t)
            t_f_new = f_new[-3:]
            loss_dmvkd = 0.
            indices = torch.linspace(0, 1, steps=len(t_f_new))
            weights = 0.1 + (1 - 0.1) * indices
            length = len(t_f_new)
            for i in range(length):
                # weight = 1 / (10 ** (length - i - 1))
                loss_dmvkd += weights[i] * F.mse_loss(f_s, t_f_new[i])
            loss_feat = self.feat_loss_weight * F.mse_loss(f_s, f_t) + self.infer_weight * loss_dmvkd
        else:
            # diffusion 训练过程, 首先通过q_sample (也就是这里的self.prepare_diffusion_concat(f_t))得到对原始图像加噪之后的结果
            # q_sample的过程, 这里f_t可看成x_0, 即要恢复的图像(feature).d_f_t即x_t, 也就是从x_0采样得到的x_t
            # 公示$x_t = \sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon_t$
            # 准备采样步数 t, 根据t生成的不同噪声 noise, 以及采样得到的x_t
            f_x_t, noise, t = self.prepare_diffusion_concat(f_t)
            pred_t_noise = self.rec_module(f_x_t, t, f_t) # pred为预测的噪声 $\hat{\epsilon}_{\theta}(x_t, t)$
            loss_ddim = F.mse_loss(pred_t_noise, noise)
            loss_feat = self.rec_weight * loss_ddim + self.feat_loss_weight * F.mse_loss(f_s, f_t)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": torch.tensor(loss_feat, dtype=torch.float32, device=loss_ce.device),
        }
        return logits_student, losses_dict

    # def prepare_noise_feature(self, feature):
    #     d_feature, d_noise, d_t = self.prepare_diffusion_concat(feature)

    def prepare_diffusion_concat(self, feature):
        t = torch.randint(0, self.num_timesteps, (1,), device=feature.device).long()
        noise = torch.randn_like(feature, device=feature.device)

        x_start = feature
        x_start = (x_start * 2. - 1.) * self.d_scale

        # noise sample
        # $x_t = \sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon_t$
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x = torch.clamp(x, min=-1 * self.d_scale, max=self.d_scale)
        x = (x / self.d_scale + 1.) / 2.

        return x, noise, t

    # q_sample 计算x_t, 使用网络对噪声进行预测, 之后计算和真实噪声之间的误差
    def q_sample(self, x_start, t, noise):
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def ddim_sampling(self, f_t, cond=None, x_T=None,
                      ddim_use_original_steps=False,
                      callback=None, x0=None, img_callback=None, log_every_t=100):
        device = f_t.device
        shape = f_t.shape
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        timesteps = self.ddim_timesteps

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM sampling with {total_steps} steps")

        iterator = tqdm(time_range, desc='DDIM sampling', total=total_steps)
        D_fss = []
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(img, ts, cond, index=index)
            img, pred_x0 = outs
            D_fss.append(pred_x0)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
        D_fss.append(img)
        return D_fss, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, t, c, index, repeat_noise=False, temperature=1.):
        b, *_, device = *x.shape, x.device

        e_t = self.rec_module(x, t, f_t) # 模型对当前噪声状态的评估

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        #*************************
        sigmas = self.ddim_sigmas # sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
        #*************************
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        # x 是输入的随机噪声, 通过这个随机噪声以及 预测的对应到生成最终x_0 的noise e_t, 便可求得最终预测出来的x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # direction pointing to x_t
        # dir_xt 是指向下一个时间步的预测数据（x_t）的方向。
        # a_prev 是前一个时间步的扩散系数。
        # sigma_t 是当前步骤的噪声标准差。
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t # 计算在移除预测噪声后，数据需要移动多远才能到达下一个时间步的合理状态。
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature # 下一个时间步生成噪声
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise # 结合了上述所有计算，以产生下一个时间步的数据状态。
        return x_prev, pred_x0

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps,verbose=verbose)
        alphas_cumprod = self.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).cuda()

        self.alphas_cumprod = to_torch(self.alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(self.alphas_cumprod_prev)
        self.betas = to_torch(self.betas)
        # self.register_buffer('betas', to_torch(self.betas))
        # self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        # self.register_buffer('alphas_cumprod_prev', to_torch(self.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', torch.tensor(ddim_alphas_prev, dtype=torch.float32))
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)



def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def exists(x):
    return x is not None

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev