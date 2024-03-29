import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
import torch.nn.functional as F
from mdistiller.engine.diffusion_utiles import cosine_beta_schedule, default, extract
from mdistiller.engine.mvkd_utils import Model
from mdistiller.distillers.CRD import Embed, ContrastMemory, ContrastLoss


class MVKD_CRD(Distiller):

    def __init__(self, student, teacher, cfg, num_data):
        super(MVKD_CRD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.MVKD.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.MVKD.LOSS.FEAT_WEIGHT
        self.hint_layer = cfg.MVKD.HINT_LAYER
        self.rec_weight = cfg.MVKD.LOSS.REC_WEIGHT
        self.mvkd_weight = cfg.MVKD.LOSS.INFER_WEIGHT
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )

        # build diffusion
        self.class_num = cfg.MVKD.CLASS_NUM

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.MVKD.DIFFUSION.SAMPLE_STEP
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

        t_b, t_c, t_w, t_h = feat_t_shapes[self.hint_layer]
        self.use_condition = cfg.MVKD.DIFFUSION.USE_CONDITION
        self.rec_module = Model(ch=t_c, out_ch=t_c, ch_mult=(1, 2, 4), num_res_blocks=2, attn_resolutions=[t_w],
                                in_channels=t_c, resolution=t_w, dropout=0.0, use_condition=self.use_condition, class_num=self.class_num)

        # CRD config
        self.init_crd_modules(
            feat_s_shapes[self.hint_layer][1],
            feat_t_shapes[self.hint_layer][1],
            int(feat_t_shapes[self.hint_layer][1] / 2),
            num_data,
            cfg.CRD.NCE.K,
            cfg.CRD.NCE.MOMENTUM,
            cfg.CRD.NCE.TEMPERATURE,
        )

    def init_crd_modules(
            self,
            feat_s_channel,
            feat_t_channel,
            feat_dim,
            num_data,
            k=16384,
            momentum=0.5,
            temperature=0.07,
    ):
        self.embed_s = Embed(feat_s_channel, feat_dim)
        self.embed_t = Embed(feat_t_channel, feat_dim)
        self.contrast = ContrastMemory(feat_dim, num_data, k, temperature, momentum)
        self.criterion_s = ContrastLoss(num_data)
        self.criterion_t = ContrastLoss(num_data)

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(
            self.rec_module.parameters()) + list(self.embed_s.parameters()) + list(self.embed_t.parameters())

    def get_extra_parameters(self):
        params = (
                list(self.rec_module.parameters())
                + list(self.embed_s.parameters())
                + list(self.embed_t.parameters())
        )
        num_p = 0
        for p in params:
            num_p += p.numel()
        return num_p

    def crd_loss(self, f_s, f_t, idx, contrast_idx):
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        return s_loss + t_loss

    def forward_train(self, image, target, index, contrastive_index, **kwargs):
        cur_epoch = kwargs['epoch']
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        f_t = feature_teacher["feats"][self.hint_layer]
        b, c, h, w = f_t.shape
        if cur_epoch > self.first_rec_kd:
            loss_kd = 0.
            for i in range(self.diff_num):
                diffusion_f_t = self.ddim_sample(f_t, conditional=logits_teacher) if self.use_condition else self.ddim_sample(f_t)
                loss_kd += self.rec_weight * self.crd_loss(
                    feature_student["pooled_feat"],
                    self.teacher.avgpool(diffusion_f_t).view(b, -1),
                    index,
                    contrastive_index,
                )
            loss_kd /= self.diff_num
        else:
            x_feature_t, noise, t = self.prepare_diffusion_concat(f_t)
            rec_feature_t = self.rec_module(x=x_feature_t.float(), t=t,
                                            conditional=logits_teacher) if self.use_condition else self.rec_module(
                x_feature_t.float(), t)
            rec_loss = self.rec_weight * F.mse_loss(rec_feature_t, f_t)
            fitnet_loss = self.feat_loss_weight * self.crd_loss(
                feature_student["pooled_feat"],
                feature_teacher["pooled_feat"],
                index,
                contrastive_index,
            )
            loss_kd = rec_loss + fitnet_loss

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
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
    def ddim_sample(self, feature, conditional=None):
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
                pred_noise, x_start = self.model_predictions(f.float(), time_cond, conditional)
            else:
                pred_noise, x_start = self.model_predictions(f.float(), time_cond)

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

    def model_predictions(self, f, t, conditional=None):
        x_f = torch.clamp(f, min=-1 * self.scale, max=self.scale)
        x_f = ((x_f / self.scale) + 1.) / 2.
        if conditional is not None:
            pred_f = self.rec_module(x=x_f, t=t, conditional=conditional)
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
