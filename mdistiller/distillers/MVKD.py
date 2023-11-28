import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
from mdistiller.engine.mvkd_utils import Model


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
        self.rec_weight = cfg.MVKD.LOSS.REC_WEIGHT
        timesteps = 1000
        sampling_timesteps = cfg.MVKD.NUM_TIMESTEPS
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.d_scale = cfg.MVKD.D_SCALE
        self.f_t_shapes = feat_t_shapes
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        t_b, t_c, t_w, t_h = feat_t_shapes[self.hint_layer]
        self.rec_module = Model(ch=t_c, out_ch=t_c, num_res_blocks=2, attn_resolutions=[4], in_channels=t_c, resolution=t_w, dropout=0.0)
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

        if cur_epoch > 0:
            f_new = self.ddim_sample(f_t)
            t_f_new = f_new[-3:]
            loss_feat = 0.
            indices = torch.linspace(0, 1, steps=len(t_f_new))
            weights = 0.1 + (1 - 0.1) * indices
            length = len(t_f_new)
            for i in range(length):
                # weight = 1 / (10 ** (length - i - 1))
                loss_feat += weights[i] * F.mse_loss(f_s, t_f_new[i])
            loss_feat += F.mse_loss(f_s, f_t)
            loss_feat = 0.1 * loss_feat
        else:
            d_f_t, noise, t = self.prepare_diffusion_concat(f_t)
            d_f_t = self.rec_module(d_f_t, t)
            loss_feat = self.feat_loss_weight * F.mse_loss(
                d_f_t, f_t
            ) + self.feat_loss_weight * F.mse_loss(f_s, f_t)
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
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x = torch.clamp(x, min=-1 * self.d_scale, max=self.d_scale)
        x = (x / self.d_scale + 1.) / 2.

        return x, noise, t

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
    def ddim_sample(self, feature, clip_denoised=True):
        batch = feature.shape[0]
        shape = self.f_t_shapes
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        fully_f = []
        f = torch.randn_like(feature, device=feature.device)
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=feature.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            x_start = self.rec_module(feature, time_cond)
            x_start = (x_start * 2. - 1.) * self.d_scale
            x_start = torch.clamp(x_start, min=-1 * self.d_scale, max=self.d_scale)
            pred_noise = self.predict_noise_from_start(f, time_cond, x_start)
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.rand_like(f)

            f = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            fully_f.append(f)
        return fully_f



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
