import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
from mdistiller.engine.kd_loss import KDQualityFocalLoss, kd_loss, dkd_loss


from mdistiller.engine.transformer_utils import MultiHeadAttention
from torch.nn import Transformer


class MV1(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(MV1, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = 1
        self.beta = 8

        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

        self.hint_layer = -1
        self.mask_per = 0.2

        # self.conv_reg = MultiHeadAttention(256, 4)
        self.conv_reg = Transformer(d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024)
        # feat_s_shapes, feat_t_shapes = get_feat_shapes(
        #     self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        # )
        # self.conv_reg = ConvReg(
        #     feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        # )

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
        # 1. cls loss
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # 2. KD loss
        b, c, h, w = feature_student["feats"][self.hint_layer].shape
        f_s = feature_student["feats"][self.hint_layer].reshape(b, c, -1).transpose(2, 1).contiguous()
        f_t = feature_teacher["feats"][self.hint_layer].reshape(b, c, -1).transpose(2, 1).contiguous()
        f_cross = self.conv_reg(f_s, f_t)

        f_cross = f_cross.transpose(2,1).reshape(b,c,h,w).contiguous()

        # weight_map = F.adaptive_avg_pool1d(nn.AvgPool2d(h)(f_s_w).reshape(b, -1))
        # sorted_indices = torch.argsort(weight_map, dim=1)
        # sorted_length = int(weight_map.shape[1] * self.mask_per)
        # top_indices = sorted_indices[:, : sorted_length]
        # mask = torch.zeros_like(weight_map).scatter_(1, top_indices, 1).bool()

        # CrossKD
        kd_logits_s = self.teacher.fc(nn.AvgPool2d(h)(f_cross).reshape(b, -1))
        loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            kd_logits_s,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
