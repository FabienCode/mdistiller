import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

from mdistiller.engine.transformer_utils import MultiHeadAttention


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature, mask=None):
    if mask is None:
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)
    else:
        gt_mask = mask
        other_mask = ~mask
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
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

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class MV1(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(MV1, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

        # attention
        self.mask_attention = MultiHeadAttention(256, 4)


    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        
        ###### Add module
        b, c, h, w = feature_teacher["feats"][-1].size()
        s_feat_trans = feature_student["feats"][-1].reshape(b, c, -1).transpose(2, 1).contiguous()

        attention_feat, attention_map = self.mask_attention(s_feat_trans, s_feat_trans, s_feat_trans)

        # kd_logits = self.student.fc(nn.AvgPool1d(h*w)(attention_feat).reshape(b, -1))
        # kd_weight = self.student.fc(nn.AvgPool1d(h*w)(attention_map).reshape(b, -1))
        # sorted_indices = torch.argsort(kd_weight, dim=1)
        # top_length = int(kd_weight.shape[-1] * 0.5)
        # top_indices = sorted_indices[:, : top_length]
        # kd_mask = torch.ones_like(kd_weight).scatter_(1, top_indices, 0).bool()
        loss_kd = F.mse_loss(attention_feat.reshape(b,c,h,w), feature_teacher["feats"][-1])
        ###### END Add
        # KD loss
        # loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
        #     kd_logits,
        #     logits_teacher,
        #     target,
        #     self.alpha,
        #     self.beta,
        #     self.temperature,
        #     kd_mask
        # )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

    def get_learnable_parameters(self):

        return [v for k, v in self.student.named_parameters()] + \
            list(self.mask_attention.parameters())

def print_grad(grad):
    print(grad)
