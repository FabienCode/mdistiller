import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes

from mdistiller.engine.area_utils import AreaDetection, extract_regions
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

        self.area_num = cfg.RegKD.AREA_NUM
        self.hint_layer = cfg.RegKD.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )

        self.area_det = AreaDetection(int(feat_t_shapes[-1][1]), int(feat_t_shapes[-1][1]), 2)

        self.channel_mask = cfg.RegKD.CHANNEL_MASK

    # list(self.score_norm.parameters())
    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.area_det.parameters()) + list(self.conv_reg.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.area_det.parameters():
            num_p += p.numel()
        return num_p
        
    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # KD loss
        # 1. DKD loss
        if 'vgg' not in self.cfg.DISTILLER.STUDENT:
            fc_mask = prune_fc_layer(self.teacher.fc, self.channel_mask).unsqueeze(0).expand(logits_student.shape[0], -1).cuda()
        else:
            fc_mask = prune_fc_layer(self.teacher.classifier, self.channel_mask).unsqueeze(0).expand(logits_student.shape[0], -1).cuda()
        #  min(kwargs["epoch"] / sel.warmup, 1.0) *
        loss_dkd = self.channel_weight * min(kwargs["epoch"] / self.warmup, 1.0) * mask_logits_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            fc_mask,
        )
        # 2. RegKD loss
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        f_t = feature_teacher["feats"][self.hint_layer]
        heat_map, wh, offset = self.area_det(f_t)
        masks, scores = extract_regions(f_t, heat_map, wh, offset, self.area_num, 3)

        loss_regkd = self.area_weight * min(kwargs["epoch"] / self.warmup, 1.0) * aaloss(f_s, f_t, masks, scores)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
            "losses_reg": loss_regkd
        }
        return logits_student, losses_dict

def aaloss(feature_student,
           feature_teacher,
           masks,
           scores):
    masks_stack = torch.stack(masks)
    scores_expand = scores.unsqueeze(-1).unsqueeze(-1).expand_as(masks_stack)
    weight_masks = masks_stack * scores_expand
    s_masks = weight_masks.sum(-1)
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



