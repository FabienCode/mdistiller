import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
from mdistiller.engine.area_utils import AreaDetection, extract_regions

import time

class MV1(Distiller):
    """Automaticv importance area detection for Knowledge distillation"""

    def __init__(self, student, teacher, cfg):
        super(MV1, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = 1
        self.beta = 8

        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

        self.hint_layer = -1
        self.mask_per = 0.2
        self.conv_reg = AreaDetection(256, 256, 2)
        self.area_num = 8
        self.score_norm = nn.BatchNorm1d(self.area_num)
        self.score_relu = nn.ReLU()

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters())

# tese gitte key4
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
        # 1. cls loss
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # 2. KD loss
        f_s = feature_student["feats"][self.hint_layer]
        f_t = feature_teacher["feats"][self.hint_layer]
        b, c, h, w = f_s.shape
        heat_map, wh, offset = self.conv_reg(f_s)
        masks, scores = extract_regions(f_s, heat_map, wh, offset, self.area_num, 3)
        # scores = norm_tensor(scores)
        scores = self.score_relu(self.score_norm(scores))

        aaloss_weight = 3
        # * min(kwargs["epoch"] / self.warmup, 1.0)
        loss_kd = aaloss_weight * aaloss(f_s, f_t, masks, scores)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

# test vscode
def aaloss(feature_student,
           feature_teacher,
           masks,
           scores):
    loss = 0
    # scores = F.normalize(scores, p=2, dim=-1)
    for i in range(len(masks)):
        for j in range(masks[i].shape[0]):
            loss += scores[i][j] * F.mse_loss(feature_student*(masks[i][j].unsqueeze(0).unsqueeze(0)), feature_teacher*(masks[i][j].unsqueeze(0).unsqueeze(0)))

    return loss

def norm_tensor(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)
    
