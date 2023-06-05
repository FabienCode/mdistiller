import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

# Decoupled Knowledge Distillation(CVPR 2022)
def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
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


def fitnet_loss(feature_s, feature_t, weight):
    return weight * F.mse_loss(feature_s, feature_t)

#   Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
def single_stage_at_loss(f_s, f_t, p):
    def _at(feat, p):
        return F.normalize(feat.pow(p).mean(1).reshape(feat.size(0), -1))

    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    return (_at(f_s, p) - _at(f_t, p)).pow(2).mean()

def at_loss(g_s, g_t, p):
    return sum([single_stage_at_loss(f_s, f_t, p) for f_s, f_t in zip(g_s, g_t)])
##### AT end #####

# Distilling the Knowledge in a Neural Network
def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd
#### kd end #####
def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

# RKD Relational Knowledge Disitllation, CVPR2019
def _pdist(e, squared, eps):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()
    
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

def rkd_loss(f_s, f_t, squared=False, eps=1e-12, distance_weight=25, angle_weight=50):
    stu = f_s.view(f_s.shape[0], -1)
    tea = f_t.view(f_t.shape[0], -1)

    # RKD distance loss
    with torch.no_grad():
        t_d = _pdist(tea, squared, eps)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td

    d = _pdist(stu, squared, eps)
    mean_d = d[d > 0].mean()
    d = d / mean_d

    loss_d = F.smooth_l1_loss(d, t_d)

    # RKD Angle loss
    with torch.no_grad():
        td = tea.unsqueeze(0) - tea.unsqueeze(1)
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = stu.unsqueeze(0) - stu.unsqueeze(1)
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss_a = F.smooth_l1_loss(s_angle, t_angle)

    loss = distance_weight * loss_d + angle_weight * loss_a
    return loss


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class MDis(Distiller):

    def __init__(self, student, teacher, cfg):
        super(MDis, self).__init__(student, teacher)

        # DKD super-parameters setting
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP
        self.kd_temperature = cfg.KD.TEMPERATURE
        # RKD setting
        self.rkd_ce_loss_weight = cfg.RKD.LOSS.CE_WEIGHT
        self.rkd_feat_loss_weight = cfg.RKD.LOSS.FEAT_WEIGHT
        self.rkd_squared = cfg.RKD.PDIST.SQUARED
        self.rkd_eps = cfg.RKD.PDIST.EPSILON
        self.rkd_distance_weight = cfg.RKD.DISTANCE_WEIGHT
        self.rkd_angle_weight = cfg.RKD.ANGLE_WEIGHT

        # AT super-parameters setting
        self.p = cfg.AT.P
        self.at_loss_weight = cfg.AT.LOSS.FEAT_WEIGHT
        # self.weight_at = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.weight_dkd = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # ! multi KD losses ! Begin #######
        loss_ori_kd = kd_loss(logits_student, logits_teacher, self.kd_temperature)
        # at loss
        loss_at = self.at_loss_weight * at_loss(
            feature_student["feats"][1:], feature_teacher["feats"][1:], self.p
        )
        loss_rkd = self.rkd_feat_loss_weight * rkd_loss(
            feature_student["pooled_feat"],
            feature_teacher["pooled_feat"],
            self.rkd_squared,
            self.rkd_eps,
            self.rkd_distance_weight,
            self.rkd_angle_weight,
        )
        # ! multi KD losses ! End #######
        kd_sum = loss_ori_kd + loss_at + loss_rkd
        loss_kd = loss_ori_kd / kd_sum * loss_ori_kd + loss_at / kd_sum * loss_at + loss_rkd / kd_sum * loss_rkd
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
