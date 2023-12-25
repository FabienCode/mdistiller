import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
import torch.nn.functional as F
from mdistiller.engine.diffusion_utiles import cosine_beta_schedule, default, extract
from mdistiller.engine.mvkd_utils import Model

from transformers import CLIPProcessor, CLIPModel
import numpy as np


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd


class MVKD(Distiller):
    """FitNets: Hints for Thin Deep Nets"""

    def __init__(self, student, teacher, cfg):
        super(MVKD, self).__init__(student, teacher)
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
        self.condition_dim = cfg.MVKD.CONDITION_DIM

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
        self.rec_module = Model(ch=t_c, out_ch=t_c, ch_mult=(1, 2), num_res_blocks=2, attn_resolutions=[t_w],
                                in_channels=t_c, resolution=t_w, dropout=0.0, use_condition=self.use_condition,
                                condition_dim=self.condition_dim)
        # self.rec_module = Model(ch=t_c*2, out_ch=t_c, ch_mult=(1, 2, 4), num_res_blocks=1, attn_resolutions=[4, 8],
        #                         in_channels=t_c*2, resolution=t_w, dropout=0.0)
        # self.proj = nn.Sequential(
        #     nn.Conv2d(t_c, t_c, 1),
        #     nn.BatchNorm2d(t_c)
        # )

        # at config
        self.p = cfg.AT.P

        # CLIP model init
        # clip_dir = os.path.join(os.getcwd(), "../", 'clip_models')
        clip_dir = os.path.join(os.getcwd(), 'clip_models')
        # clip_path = str(Path(clip_dir).resolve())
        self.clip_model = CLIPModel.from_pretrained(clip_dir).cuda()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_dir)

        self.color_token = nn.Parameter(torch.zeros(1, t_c))
        self.shape_token = nn.Parameter(torch.zeros(1, t_c))

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters()) + list(
            self.rec_module.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        for p in self.rec_module.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        cur_epoch = kwargs.get("epoch")
        device = image_weak.device
        logits_student_weak, feature_student_weak = self.student(image_weak)
        logits_student_strong, feature_student_strong = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, feature_teacher_weak = self.teacher(image_weak)
            logits_teacher_strong, feature_teacher_strong = self.teacher(image_strong)

        # losses
        batch_size, class_num = logits_student_strong.shape

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        # losses
        loss_ce = self.ce_loss_weight * (
                F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
        loss_logits = multi_loss(logits_student_weak, logits_teacher_weak,
                                 logits_student_strong, logits_teacher_strong,
                                 mask, self.ce_loss_weight)

        # loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student_weak, target)
        f_s = self.conv_reg(feature_student_weak["feats"][self.hint_layer])
        f_t = feature_teacher_weak["feats"][self.hint_layer]

        # MVKD loss
        b, c, h, w = f_t.shape
        temp_text = 'A new reconstructed feature map of '
        code_tmp = []
        for i in range(b):
            article = determine_article(CIFAR100_Labels[target[i].item()])
            color_choice = COLORS[torch.randint(0, len(COLORS), (1,)).item()]
            size_choice = SIZES[torch.randint(0, len(SIZES), (1,)).item()]

            # A reconstructed feature map of a medium-sized, red turtle
            # code_tmp.append(temp_text + article + " " + CIFAR100_Labels[target[i].item()] + '.')
            code_tmp.append(temp_text + size_choice + ", " + color_choice + " " + CIFAR100_Labels[target[i].item()])
        with torch.no_grad():
            code_inputs = self.clip_processor(text=code_tmp, return_tensors="pt", padding=True).to(device)
            context_embd = self.clip_model.get_text_features(**code_inputs)
        diff_con = torch.concat((context_embd, logits_teacher_strong), dim=-1)
        # for i in range(b):
        #
        #     # A reconstructed feature map of a medium-sized, red turtle
        #     # code_tmp.append(temp_text + article + " " + CIFAR100_Labels[target[i].item()] + '.')
        #     code_tmp.append(temp_text + CIFAR100_Labels[target[i].item()])
        # with torch.no_grad():
        #     code_inputs = self.clip_processor(text=code_tmp, return_tensors="pt", padding=True).to(device)
        #     context_embd = self.clip_model.get_text_features(**code_inputs)
        # shape_token = self.shape_token.expand(b, -1)
        # color_token = self.color_token.expand(b, -1)
        # diff_con = torch.concat((context_embd, shape_token, color_token, logits_teacher_strong), dim=-1)
        # diff_con = context_embd

        # if cur_epoch > self.first_rec_kd:
        # if cur_epoch % 2 == 1:
        mvkd_loss = 0.
        # diffusion_f_t = 0.
        for i in range(self.diff_num):
            diffusion_f_t = self.ddim_sample(f_t, conditional=diff_con) if self.use_condition else self.ddim_sample(
                f_t)
            mvkd_loss += F.mse_loss(f_s, diffusion_f_t)

        loss_kd_infer = self.mvkd_weight * mvkd_loss
        # loss_kd = self.mvkd_weight * mvkd_loss + self.feat_loss_weight * F.mse_loss(f_s, f_t)
        #
        # else:
        x_feature_t, noise, t = self.prepare_diffusion_concat(f_t)
        rec_feature_t = self.rec_module(x=x_feature_t.float(), t=t,
                                        conditional=diff_con) if self.use_condition else self.rec_module(
            x_feature_t.float(), t)
        rec_loss = self.rec_weight * F.mse_loss(rec_feature_t, f_t)
        fitnet_loss = self.feat_loss_weight * F.mse_loss(f_s, f_t)
        loss_kd_train = rec_loss + fitnet_loss
        # loss_kd = rec_loss + fitnet_loss
        loss_kd = loss_kd_train + loss_kd_infer
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_logits": loss_logits
        }
        return logits_student_weak, losses_dict

    # def prepare_targets(self, feature):
    #     d_feature, d_noise, d_t = self.prepare_diffusion_concat(feature)
    #     return d_feature, d_noise, d_t

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


def single_stage_at_loss(f_s, f_t, p):
    def _at(feat, p):
        return F.normalize(feat.pow(p).mean(1).reshape(feat.size(0), -1))

    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    return (_at(f_s, p) - _at(f_t, p)).pow(2).mean()


def at_loss(g_s, g_t, p):
    return sum([single_stage_at_loss(f_s, f_t, p) for f_s, f_t in zip(g_s, g_t)])


def multi_loss(logits_student_weak, logits_teacher_weak,
               logits_student_strong, logits_teacher_strong,
               mask, weight):
    loss_kd_weak = (weight * ((kd_loss(logits_student_weak, logits_teacher_weak, 4) * mask).mean()))

    # loss_kd_weak = (weight * ((kd_loss(logits_student_weak, logits_teacher_weak, 4) * mask).mean()))

    loss_kd_strong = (weight * ((kd_loss(logits_student_strong, logits_teacher_strong, 4) * mask).mean()))

    loss_cc_weak = (weight * ((cc_loss(logits_student_weak, logits_teacher_weak, 4) * mask).mean()))
    loss_cc_strong = (weight * ((cc_loss(logits_student_strong, logits_teacher_strong, 4) * mask).mean()))
    loss_cc = loss_cc_weak + loss_cc_strong

    loss_bc_weak = (weight * ((bc_loss(logits_student_weak, logits_teacher_weak, 4) * mask).mean()))
    loss_bc_strong = (weight * ((bc_loss(logits_student_strong, logits_teacher_strong, 4) * mask).mean()))
    loss_bc = loss_bc_weak + loss_bc_strong

    return loss_kd_weak + loss_kd_strong + loss_cc_weak + loss_bc_weak


def determine_article(word):
    """根据单词的首字母确定使用 'a' 还是 'an'。"""
    vowels = "aeiou"
    return "an" if word[0].lower() in vowels else "a"


CIFAR100_Labels = {
    0: "apple", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver",
    5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottle",
    10: "bowl", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
    15: "camel", 16: "can", 17: "castle", 18: "caterpillar", 19: "cattle",
    20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
    25: "couch", 26: "crab", 27: "crocodile", 28: "cup", 29: "dinosaur",
    30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
    35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard",
    40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
    45: "lobster", 46: "man", 47: "maple_tree", 48: "motorcycle", 49: "mountain",
    50: "mouse", 51: "mushroom", 52: "oak_tree", 53: "orange", 54: "orchid",
    55: "otter", 56: "palm_tree", 57: "pear", 58: "pickup_truck", 59: "pine_tree",
    60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum",
    65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
    70: "rose", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
    75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
    80: "squirrel", 81: "streetcar", 82: "sunflower", 83: "sweet_pepper", 84: "table",
    85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor",
    90: "train", 91: "trout", 92: "tulip", 93: "turtle", 94: "wardrobe",
    95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman", 99: "worm"
}

COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "black", "white", "grey", "pink"]
SIZES = ["small", "large", "tiny", "big", "huge"]


def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss
