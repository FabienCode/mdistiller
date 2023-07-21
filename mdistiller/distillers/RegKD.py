import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

from mdistiller.engine.area_utils import AreaDetection, extract_regions

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
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
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
            "losses_dict": loss_regkd
        }
        return logits_student, losses_dict

def aaloss(feature_student,
           feature_teacher,
           masks,
           scores):
    loss = 0
    scores = F.normalize(scores, p=2, dim=-1)
    for i in range(len(masks)):
        for j in range(masks[i].shape[0]):
            loss += scores[i][j] * F.mse_loss(feature_student*(masks[i][j].unsqueeze(0).unsqueeze(0)), feature_teacher*(masks[i][j].unsqueeze(0).unsqueeze(0)))

    return loss

def norm_tensor(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)
