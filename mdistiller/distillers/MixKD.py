import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd


class MixKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(MixKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.FITNET.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT
        self.hint_layer = cfg.FITNET.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )
        self.beta = 1.0
        self.cutmix_prob = 0.5

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, feature_student_weak = self.student(image_weak)
        # logits_student_strong, feature_student_strong = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, feature_teacher_weak = self.teacher(image_weak)
            logits_teacher_strong, feature_teacher_strong = self.teacher(image_strong)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student_weak, target)
        # f_s_weak = self.conv_reg(feature_student_weak["feats"][self.hint_layer])

        # f_t_weak = feature_teacher_weak["feats"][self.hint_layer]
        # f_t_strong = feature_teacher_strong["feats"][self.hint_layer]
        # s_t_weak = calculate_saliency(f_t_weak)
        # s_t_strong = calculate_saliency(f_t_strong)
        # mix_t_weak, mix_t_strong = mix_features(f_t_weak, f_t_strong, s_t_weak, s_t_strong)

        # loss_feat = F.mse_loss(f_s_weak, mix_t_weak) + F.mse_loss(f_s_weak, mix_t_strong)
        loss_kd = kd_loss(logits_student_weak, logits_teacher_weak, 4) + kd_loss(
            logits_student_weak, logits_teacher_strong, 4
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_feat_weak": loss_kd
        }
        return logits_student_weak, losses_dict


def calculate_saliency(feature_map):
    saliency = F.adaptive_avg_pool2d(feature_map, 1)
    return saliency

def mix_features(f_a, f_b, s_a, s_b):
    weights_a = s_a / (s_a + s_b)
    weights_b = s_b / (s_a + s_b)

    w_a_expand = weights_a.expand_as(f_a)
    w_b_expand = weights_b.expand_as(f_b)

    mix_f_a = w_a_expand * f_a + (1 - w_a_expand) * f_b
    mix_f_b = w_b_expand * f_b + (1 - w_b_expand) * f_a
    return mix_f_a, mix_f_b

# def mix_feature(x, y, alpha):
#     if alpha > 0.:
#         beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
#         lma = beta_distribution.sample()
#     else:
#         lam = 1.
#
#     bbx1, bby1, bbx2, bby2 = rand_bbox(f_w.size(), lma)
#     f_w[:, :, bbx1:bbx2, bby1:bby2] = f_s[:, :, bbx1:bbx2, bby1:bby2]
#     return f_w, lma
#
#
# def rand_bbox(size, lam):
#     W = size[2]
#     H = size[3]
#     cut_rat = torch.sqrt(1. - lam)
#     cut_w = torch.tensor(W, dtype=torch.float32).mul(cut_rat).type(torch.int)
#     cut_h = torch.tensor(H, dtype=torch.float32).mul(cut_rat).type(torch.int)
#
#     # uniform
#     cx = torch.randint(0, W, (1,)).item()
#     cy = torch.randint(0, H, (1,)).item()
#
#     bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
#     bby1 = torch.clamp(cy - cut_h // 2, 0, H)
#     bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
#     bby2 = torch.clamp(cy + cut_h // 2, 0, H)
#
#     return bbx1, bby1, bbx2, bby2
