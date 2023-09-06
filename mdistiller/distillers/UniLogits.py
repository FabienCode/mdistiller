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
    loss_kd *= temperature**2
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

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.logits_weight * kd_loss(
            logits_student, logits_teacher, self.temperature
        )

        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        f_t = feature_teacher["feats"][self.hint_layer]
        loss_feat = self.feat_weight * feature_dis_loss(f_s, f_t, self.temperature)
        # loss_feat = min(kwargs["epoch"] / self.warmup, 1.0) * self.channel_weight * \
        #         feature_dkd_dis_loss(f_s, f_t, target, self.alpha, self.beta, self.temperature)
        # loss_feat = 100 * F.mse_loss(
        #     f_s, feature_teacher["feats"][self.hint_layer]
        # )

        # loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
        #     logits_student,
        #     logits_teacher,
        #     target,
        #     self.alpha,
        #     self.beta,
        #     self.temperature,
        # )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_feature": loss_feat,
            "loss_dkd": loss_kd
        }
        return logits_student, losses_dict
