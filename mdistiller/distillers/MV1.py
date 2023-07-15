import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
from mdistiller.engine.area_utils import AreaDetection, extract_regions


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
        f_s = feature_student["feats"][self.hint_layer]
        f_t = feature_teacher["feats"][self.hint_layer]
        b, c, h, w = f_s.shape
        heat_map, wh, offset = self.conv_reg(feature_student["feats"][self.hint_layer])
        # loss_kd = F.mse_loss(f_s, f_t)
        loss_kd = aaloss(f_s, f_t, heat_map, wh, offset, k=3, kernel=3)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

# test pycharm
def aaloss(feature_student,
           feature_teacher,
           center_heat_map,
           wh_pred,
           offset_pred,
           k=20,
           kernel=3):
    loss = 0
    masks = extract_regions(feature_student, center_heat_map, wh_pred, offset_pred, k, kernel)
    #
    for i in range(len(masks)):
        for j in range(masks[i].shape[0]):
            loss += F.mse_loss(feature_student*(masks[i][j].unsqueeze(0).unsqueeze(0)), feature_teacher*(masks[i][j].unsqueeze(0).unsqueeze(0)))
    # new_masks = torch.stack(masks)
    # batch_size, num_masks = new_masks.shape[0], new_masks.shape[1]
    # b, c, h, w = feature_student.size()
    # # masks = torch.stack(masks, dim=1)  # new masks shape: [batch_size, num_masks, height, width]
    #
    # # expand features to match masks shape
    # feature_student = feature_student.unsqueeze(1).expand(-1, new_masks.shape[1], -1, -1, -1)
    # feature_teacher = feature_teacher.unsqueeze(1).expand(-1, new_masks.shape[1], -1, -1, -1)
    #
    # # apply masks
    # masked_feature_student = feature_student * new_masks.unsqueeze(2)
    # masked_feature_teacher = feature_teacher * new_masks.unsqueeze(2)
    #
    # # compute MSE loss
    # loss = F.mse_loss(masked_feature_student, masked_feature_teacher)

    return loss
    
