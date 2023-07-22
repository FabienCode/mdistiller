import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

from mdistiller.engine.area_utils import AreaDetection, extract_regions
from mdistiller.engine.kd_loss import mask_logits_loss
from mdistiller.models.cifar.resnet import BasicBlock, Bottleneck

class RegKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(RegKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

        self.area_num = 8
        self.hint_layer = -1

        self.area_det = AreaDetection(256, 256, 2)

        self.channel_mask = 0.9

    # list(self.score_norm.parameters())
    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.area_det.parameters())

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
        fc_mask = prune_fc_layer(self.student.fc, self.channel_mask).unsqueeze(0).expand(logits_student.shape[0], -1).cuda()
        #  min(kwargs["epoch"] / self.warmup, 1.0) *
        loss_dkd = 3 * mask_logits_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            fc_mask,
        )
        # 2. RegKD loss
        f_s = feature_student["feats"][self.hint_layer]
        f_t = feature_teacher["feats"][self.hint_layer]
        heat_map, wh, offset = self.area_det(f_s)
        masks, scores = extract_regions(f_s, heat_map, wh, offset, self.area_num, 3)

        regloss_weight = 3
        loss_regkd = regloss_weight * aaloss(f_s, f_t, masks, scores)
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
    loss = 0
    scores = F.normalize(scores, p=1, dim=1)
    for i in range(len(masks)):
        for j in range(masks[i].shape[0]):
            loss += scores[i][j] * F.mse_loss(feature_student*(masks[i][j].unsqueeze(0).unsqueeze(0)), feature_teacher*(masks[i][j].unsqueeze(0).unsqueeze(0)))

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



