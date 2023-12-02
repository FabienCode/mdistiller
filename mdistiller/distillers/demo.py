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


class KD_strong(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD_strong, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.FITNET.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT
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

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, feature_student_weak = self.student(image_weak)
        logits_student_strong, feature_student_strong = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, feature_teacher_weak = self.teacher(image_weak)
            logits_teacher_strong, feature_teacher_strong = self.teacher(image_strong)

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
        f_s_weak = self.conv_reg(feature_student_weak["feats"][self.hint_layer])
        loss_feat_weak = self.feat_loss_weight * F.mse_loss(
            f_s_weak, feature_teacher_weak["feats"][self.hint_layer]
        )
        f_s_strong = self.conv_reg(feature_student_strong["feats"][self.hint_layer])
        loss_feat_strong = self.feat_loss_weight * F.mse_loss(
            f_s_strong, feature_teacher_strong["feats"][self.hint_layer]
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_feat_weak": loss_feat_weak,
            "loss_feat_strong": loss_feat_strong,
        }
        return logits_student_weak, losses_dict
