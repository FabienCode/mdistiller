import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
from mdistiller.engine.kd_loss import KDQualityFocalLoss, kd_loss, dkd_loss


from mdistiller.engine.transformer_utils import MultiHeadAttention
from torch.nn import Transformer
from mdistiller.engine.area_utils import AreaDetection, extract_area


# class AutoAreaDetection(nn.Module):
#     def __init__(slef, d_model):
    


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
        # self.conv_reg = Transformer(d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024)
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        # self.conv_reg = ConvReg(
        #     feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        # )
        # self.conv_reg = ConvReg(
        #     feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        # )
        self.conv_reg = AreaDetection()

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
        heat_map, wh, offset = self.conv_reg(feature_student["feats"][self.hint_layer])


        losses_dict = {
            "loss_ce": loss_ce,
            # "loss_kd": loss_kd,
            # "loss_feat": loss_feat
        }
        return logits_student, losses_dict
    
# def AALoss(feature_student,
#            feature_teacher,
#            center_heat_map,
#            wh_pred,
#            offset_pred,
#            score_threshold=0.5,
#            resize_to_original=False):
    
