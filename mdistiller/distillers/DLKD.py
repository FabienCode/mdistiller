import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
import torch.nn.functional as F
from mdistiller.engine.diffusion_utiles import cosine_beta_schedule, default, extract
from mdistiller.engine.mvkd_utils import Model

from transformers import CLIPProcessor, CLIPModel, ViltProcessor, ViltForQuestionAnswering
import numpy as np
from mdistiller.engine.light_diffusion import DiffusionModel, AutoEncoder
from mdistiller.engine.classes_datasets import CIFAR100_Labels, Imagenet_Labels


# def kd_loss(logits_student, logits_teacher, temperature):
#     log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
#     pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
#     loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
#     loss_kd *= temperature ** 2
#     return loss_kd


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand=True):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class DLKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(DLKD, self).__init__(student, teacher)
        self.temperature = cfg.DLKD.TEMPERATURE
        self.ce_loss_weight = cfg.DLKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.DLKD.LOSS.KD_WEIGHT
        self.rec_logits_weight = cfg.DLKD.LOSS.REC_LOGITS_WEIGHT

        # diffusion dim
        self.logit_dim = cfg.DLKD.DIFF.CLS_NUM
        self.diff_num = cfg.DLKD.DIFF.DIFF_NUM

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.DLKD.DIFF.SAMPLE_STEP
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
        self.scale = cfg.MVKD.DIFFUSION.SNR_SCALE
        self.diff_num = cfg.MVKD.DIFFUSION.DIFF_FEATURE_NUM
        self.first_rec_kd = cfg.MVKD.INIT_REC_EPOCH

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

        self.use_condition = cfg.MVKD.DIFFUSION.USE_CONDITION
        # self.rec_module = Model(ch=self.logit_dim, out_ch=self.logit_dim, ch_mult=(1, 2), num_res_blocks=2, attn_resolutions=[1, 1],
        #                         in_channels=self.logit_dim, resolution=1, dropout=0.0)
        self.rec_module = DifUNet(self.logit_dim, self.logit_dim, self.logit_dim)
        # latent_dim = t_c
        # self.ae = AutoEncoder(channels=t_c, latent_channels=latent_dim)
        # # self.conv_reg = ConvReg(
        # #     feat_s_shapes[self.hint_layer], latent_dim
        # # )
        # self.conv_reg = nn.Conv2d(feat_s_shapes[self.hint_layer][1], latent_dim, 1)

        # CLIP model init
        # clip_dir = os.path.join(os.getcwd(), "../", 'clip_models')
        # clip_dir = os.path.join(os.getcwd(), 'clip_models')
        # # clip_path = str(Path(clip_dir).resolve())
        # self.clip_model = CLIPModel.from_pretrained(clip_dir).cuda()
        # self.clip_processor = CLIPProcessor.from_pretrained(clip_dir)

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.rec_module.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.rec_module.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        epoch = kwargs['epoch']
        if epoch < 150:
            epoch_weight = 0.01
        elif 150 <= epoch < 180:
            epoch_weight = 0.1
        else:
            epoch_weight = 1.
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # diffusion logit

        # train process
        x_logits_t, noise, t = self.prepare_diffusion_concat(logits_teacher)
        dif_logits_t = self.rec_module(logits_teacher, t)
        rec_loss = self.rec_logits_weight * kd_loss(dif_logits_t, logits_teacher, self.temperature)

        # diffusion logits kd
        if epoch > 150:
            logits_teacher_accum = logits_teacher.clone()  # 避免就地操作，克隆 logits_teacher
            for i in range(self.diff_num):
                dif_l_t = self.ddim_sample(logits_teacher_accum, logits_student)
                logits_teacher_accum = logits_teacher_accum + dif_l_t
                # cos_sim = F.cosine_similarity(logits_teacher_accum, dif_l_t, dim=1)
                # print("Cosine Similarity {i}:", cos_sim)
                # weighted_logits = weighted_logits + epoch_weight * dif_l_t
            normalized_logits = F.normalize(logits_teacher_accum, dim=1)
            dlkd_loss = self.kd_loss_weight * kd_loss(logits_student, normalized_logits, self.temperature)

            losses_dict = {
                "loss_ce": loss_ce,
                "rec_loss": rec_loss,
                "dlkd_loss": dlkd_loss,
            }
        else:
            loss_kd = self.kd_loss_weight * kd_loss(
                logits_student, logits_teacher, self.temperature
            )
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_kd,
                "rec_loss": rec_loss,
            }
        return logits_student, losses_dict

    def prepare_diffusion_concat(self, feature):
        t = torch.randint(0, self.num_timesteps, (1,)).cuda().long()
        noise = torch.randn_like(feature)

        x_start = feature
        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1.) / 2.

        return x, noise, t

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    @torch.no_grad()
    def ddim_sample(self, feature, s_feature, conditional=None):
        batch = feature.shape[0]
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1., total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        f = torch.randn_like(feature)
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, dtype=torch.long).cuda()
            self_cond = x_start if self.self_condition else None

            if conditional is not None:
                pred_noise, x_start = self.model_predictions(f.float(), time_cond, s_feature, conditional)
            else:
                pred_noise, x_start = self.model_predictions(f.float(), time_cond, s_feature)

            if time_next < 0:
                f = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(f)

            f = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise
        return f

    def model_predictions(self, f, t, context=None, conditional=None):
        x_f = torch.clamp(f, min=-1 * self.scale, max=self.scale)
        x_f = ((x_f / self.scale) + 1.) / 2.
        if conditional is not None:
            pred_f = self.rec_module(x=x_f, t=t, context=context, conditional=conditional)
        else:
            pred_f = self.rec_module(x_f, t)
        pred_f = (pred_f * 2 - 1.) * self.scale
        pred_f = torch.clamp(pred_f, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(f, t, pred_f)
        return pred_noise, pred_f

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )


class DifUNet(nn.Module):
    def __init__(self, in_channels, out_channels=100, t_dim=100):
        super(DifUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.time_embeddings = nn.Linear(t_dim, out_channels)

    def forward(self, x, t):
        # 检查输入的维度，如果是2D则扩展为4D
        is_1d_input = False
        if len(x.shape) == 2:  # (batch_size, feature_dim)
            is_1d_input = True
            x = x.unsqueeze(-1).unsqueeze(-1)  # 调整为 (batch_size, feature_dim, 1, 1)

        # 编码和解码
        x = self.encoder(x)
        x = self.decoder(x)
        t_embeddings = get_timestep_embedding(t, self.in_channels)
        t_embeddings = self.time_embeddings(t_embeddings)
        x = x + t_embeddings.unsqueeze(-1).unsqueeze(-1)

        # 如果输入是2D，则将输出调整回2D
        if is_1d_input:
            x = x.squeeze(-1).squeeze(-1)  # 调整为 (batch_size, feature_dim)

        return x


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb
