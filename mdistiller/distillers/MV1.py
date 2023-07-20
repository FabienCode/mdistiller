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
        self.conv_reg = AreaDetection(256, 256, 6)

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters())

# tese gitte key1
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
        aaloss_weight = 1
        # * min(kwargs["epoch"] / self.warmup, 1.0)
        loss_kd = aaloss_weight  * aaloss(f_s, f_t, heat_map, wh, offset, k=8, kernel=3)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

# test vscode
def aaloss(feature_student,
           feature_teacher,
           center_heat_map,
           wh_pred,
           offset_pred,
           k=8,
           kernel=3):
    loss = 0
    masks, scores = extract_regions(feature_student, center_heat_map, wh_pred, offset_pred, k, kernel)
    for i in range(len(masks)):
        for j in range(masks[i].shape[0]):
            loss += scores[i][j] * F.mse_loss(feature_student*(masks[i][j].unsqueeze(0).unsqueeze(0)), feature_teacher*(masks[i][j].unsqueeze(0).unsqueeze(0)))

    return loss
    
