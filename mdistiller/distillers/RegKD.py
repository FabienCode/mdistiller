import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes

from mdistiller.engine.area_utils import AreaDetection, extract_regions, RegKD_pred
from mdistiller.engine.kd_loss import mask_logits_loss, dkd_loss
from mdistiller.models.cifar.resnet import BasicBlock, Bottleneck

class RegKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(RegKD, self).__init__(student, teacher)
        self.cfg = cfg
        self.ce_loss_weight = cfg.RegKD.CE_WEIGHT
        self.alpha = cfg.RegKD.ALPHA
        self.beta = cfg.RegKD.BETA
        self.temperature = cfg.RegKD.T
        self.warmup = cfg.RegKD.WARMUP

        self.channel_weight = cfg.RegKD.CHANNEL_KD_WEIGHT
        self.area_weight = cfg.RegKD.AREA_KD_WEIGHT
        self.size_reg_weight = cfg.RegKD.SIZE_REG_WEIGHT

        self.area_num = cfg.RegKD.AREA_NUM
        self.hint_layer = cfg.RegKD.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.RegKD.INPUT_SIZE
        )
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )
        self.area_det = RegKD_pred(int(feat_t_shapes[self.hint_layer][1]),
                                   int(feat_t_shapes[self.hint_layer][1]), 2, 100, cfg.RegKD.LOGITS_THRESH)
        self.channel_mask = cfg.RegKD.CHANNEL_MASK

        # self.logits_thresh = cfg.RegKD.LOGITS_THRESH

    #
    # # list(self.score_norm.parameters())
    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.area_det.parameters()) + list(self.conv_reg.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.area_det.parameters():
            num_p += p.numel()
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        return num_p
        
    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # KD loss
        ############## !@ Reg Pred ################
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        f_t = feature_teacher["feats"][self.hint_layer]
        # f_s = self.conv_reg(feature_student["preact_feats"][self.hint_layer])
        # f_t = feature_teacher["preact_feats"][self.hint_layer]
        # heat_map, wh, offset, s_thresh, s_fc_mask = self.area_det(f_s, logits_student)
        # t_heat_map, t_wh, t_offset, t_thresh, t_fc_mask = self.area_det(f_t, logits_teacher)
        heat_map, wh, offset, s_fc_mask = self.area_det(f_s, logits_student)
        t_heat_map, t_wh, t_offset, t_fc_mask = self.area_det(f_t, logits_teacher)
        tmp_mask = s_fc_mask - t_fc_mask
        fc_mask = torch.zeros_like(tmp_mask)
        fc_mask[tmp_mask == 0] = 1
        fc_mask[s_fc_mask == 0] = 0
        fc_mask[t_fc_mask == 0] = 0
        # dis-cls loss
        loss_logits = self.channel_weight * mask_kd_loss(logits_student, logits_teacher, self.temperature, fc_mask.bool())
        b,c,h,w = heat_map.shape
        t_area = torch.cat((heat_map, wh, offset), dim=1)
        s_area = torch.cat((t_heat_map, t_wh, t_offset), dim=1)
        masks, scores = extract_regions(f_s, heat_map, wh, offset, self.area_num, 3)
        # masks, scores = extract_regions(f_t, t_heat_map, t_wh, offset, self.area_num, 3)
        # dis-feature loss
        loss_feature = self.area_weight * aaloss(f_s, f_t, masks, scores)
        # area loss
        # loss_area = self.size_reg_weight * F.mse_loss(s_area, t_area)-torch.mean(s_thresh)-torch.mean(t_thresh)
        # loss_area = self.size_reg_weight * F.mse_loss(s_area, t_area) - 0.5 * torch.sum(s_thresh**2) - 0.5 * torch.sum(t_thresh**2)
        # loss_area = self.size_reg_weight * F.mse_loss(s_area, t_area) + self.size_reg_weight * F.mse_loss(s_thresh, t_thresh)
        loss_area = self.size_reg_weight * F.mse_loss(s_area, t_area)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_dcm": loss_logits,
            "losses_reg": loss_feature,
            "losses_area": loss_area
        }
        return logits_student, losses_dict

def aaloss(feature_student,
           feature_teacher,
           masks,
           scores):
    masks_stack = torch.stack(masks)
    # scores_expand = scores.unsqueeze(-1).unsqueeze(-1).expand_as(masks_stack)
    # weight_masks = masks_stack * scores_expand
    s_masks = masks_stack.sum(1)
    loss = F.mse_loss(feature_student * s_masks.unsqueeze(1), feature_teacher * s_masks.unsqueeze(1))
    return loss

def norm_tensor(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


def calculate_channel_importance(conv_layer):
    weights = conv_layer.weight.data
    l1_norm = weights.abs().sum(dim=(1, 2, 3))
    return l1_norm


def calculate_block_importance_and_mask(model, percentile=80):
    block_importance = {}
    block_mask = {}

    for name, module in model.named_modules():
        # 判断是否是BasicBlock或Bottleneck，这两种类型的模块都包含有多个卷积层
        if isinstance(module, (BasicBlock, Bottleneck)):
            # 获取block内最后一个卷积层，对于BasicBlock是conv2，对于Bottleneck是conv3
            last_conv = module.conv2 if isinstance(module, BasicBlock) else module.conv3
            # 计算该卷积层对应的通道重要性
            importance = calculate_channel_importance(last_conv)
            # 存储结果
            block_importance[name] = importance

            # 计算阈值
            threshold = np.percentile(importance.cpu().detach().numpy(), percentile)
            # 创建掩码，如果通道的重要性大于阈值，那么掩码的值为1，否则为0
            mask = (importance > threshold).float()
            # 存储结果
            block_mask[name] = mask

    return block_importance, block_mask


def compute_fc_importance(fc_layer):
    importance = torch.norm(fc_layer.weight.data, p=1, dim=1)
    return importance

def prune_fc_layer(fc_layer, percentage):
    fc_importance = compute_fc_importance(fc_layer)
    _, sorted_index = torch.sort(fc_importance)
    num_prune = int(len(sorted_index) * percentage)
    prune_index = sorted_index[:num_prune]
    mask = torch.ones(len(sorted_index), dtype=torch.bool)
    mask[prune_index] = 0
    return mask

def mask_kd_loss(logits_student, logits_teacher, temperature, mask=None):
    if mask is not None:
        logits_student = logits_student * mask
        logits_teacher = logits_teacher * mask
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def cat_mask_kd_loss(logits_student, logits_teacher, target, temperature, alpha, beta, mask=None):
    # if mask is not None:
    #     logits_student = logits_student * mask
    #     logits_teacher = logits_teacher * mask
    gt_mask = mask
    other_mask = ~mask
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = F.kl_div(log_pred_student, pred_teacher, size_average=False) * (temperature ** 2) / target.shape[0]
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


