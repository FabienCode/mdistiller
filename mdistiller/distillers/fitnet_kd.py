import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class FitNet_KD(Distiller):
    """FitNets: Hints for Thin Deep Nets"""

    def __init__(self, student, teacher, cfg):
        super(FitNet_KD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.FITNET.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT
        self.hint_layer = cfg.FITNET.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )

        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.temperature = cfg.KD.TEMPERATURE

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
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        loss_feat = self.feat_loss_weight * F.mse_loss(
            f_s, feature_teacher["feats"][self.hint_layer]
        )
        # loss_kd = self.kd_loss_weight * kd_loss(
        #     logits_student, logits_teacher, self.temperature
        # )
        loss_kd = self.kd_loss_weight * F.mse_loss(logits_student, logits_teacher)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_feat": loss_feat,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
