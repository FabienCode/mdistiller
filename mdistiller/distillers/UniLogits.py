import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
from mdistiller.engine.kd_loss import mask_logits_loss, dkd_loss


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd


def feature_dis_loss(feature_student, feature_teacher, temperature):
    b, c, h, w = feature_student.shape
    mean_s = torch.mean(feature_student.reshape(b, c, -1), dim=1)
    mean_t = torch.mean(feature_teacher.reshape(b, c, -1), dim=1)
    # log_s = F.log_softmax(mean_s / temperature, dim=1)
    # log_t = F.softmax(mean_t / temperature, dim=1)
    # loss = F.kl_div(log_s, log_t, reduction="none").sum(1).mean()
    # loss *= temperature**2
    loss = F.mse_loss(mean_s, mean_t)
    # loss = F.kl_div(F.normalize(mean_s), F.normalize(mean_t), reduction='batchmean')
    return loss


def feature_dkd_dis_loss(feature_student, feature_teacher, target, alpha, beta, temperature):
    b, c, h, w = feature_student.shape
    mean_s = torch.mean(feature_student.reshape(b, c, -1), dim=1)
    mean_t = torch.mean(feature_teacher.reshape(b, c, -1), dim=1)
    mask = torch.zeros_like(mean_s)
    max_indices = torch.argmax(mean_t, dim=-1)
    mask[torch.arange(b), max_indices] = 1
    loss = mask_logits_loss(mean_s, mean_t, target, alpha, beta, temperature, mask.bool())
    return loss


class UniLogitsKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(UniLogitsKD, self).__init__(student, teacher)
        self.temperature = cfg.Uni.TEMPERATURE
        self.ce_loss_weight = cfg.Uni.LOSS.CE_WEIGHT
        self.logits_weight = cfg.Uni.LOSS.LOGITS_WEIGHT
        self.feat_weight = cfg.Uni.LOSS.FEAT_KD_WEIGHT
        self.supp_weight = cfg.Uni.LOSS.SUPP_WEIGHT

        # dkd para
        self.warmup = cfg.DKD.WARMUP
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA

        self.hint_layer = cfg.FITNET.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )

        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )
        self.feat2pro = featPro(feat_t_shapes[self.hint_layer][1], feat_t_shapes[self.hint_layer][2], 100)

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters()) + list(self.feat2pro.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        for p in self.feat2pro.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # loss_kd = self.logits_weight * kd_loss(
        #     logits_student, logits_teacher, self.temperature
        # )
        loss_kd = self.logits_weight * min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )

        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        f_t = feature_teacher["feats"][self.hint_layer]
        f_s_pro = self.feat2pro(f_s)
        f_t_pro = self.feat2pro(f_t)
        # loss_feat = self.feat_weight * kd_loss(f_s_pro, f_t_pro, self.temperature)
        loss_feat = self.feat_weight * F.mse_loss(f_s_pro, f_t_pro)

        loss_supp_feat2pro = self.supp_weight * \
            (kd_loss(f_s_pro, logits_student, self.temperature) + kd_loss(f_t_pro, logits_teacher, self.temperature))
        # loss_supp_feat2pro = self.supp_weight * \
        #     (F.mse_loss(f_s_pro, logits_student) + F.mse_loss(f_t_pro, logits_teacher))

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_feature": loss_feat,
            "loss_dkd": loss_kd,
            "loss_supp_feat2pro": loss_supp_feat2pro
        }
        return logits_student, losses_dict


class featPro(nn.Module):
    def __init__(self, in_channels, size, latent_dim):
        super(featPro, self).__init__()
        self.encoder = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(latent_dim * size * size, latent_dim)
        self.fc_var = nn.Linear(latent_dim * size * size, latent_dim)

    def encode(self, x):
        result = self.encoder(x)
        result = result.view(result.size(0), -1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, log_var = self.encode(x)
        # z = torch.cat((mu, log_var), dim=1)
        # z = mu + log_var
        z = self.reparameterize(mu, log_var)
        return z