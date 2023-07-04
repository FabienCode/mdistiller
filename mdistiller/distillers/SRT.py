import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from ._base import Distiller


# def taylor_approximation(feature):
#     gradient, = torch.autograd.grad(feature.sum(), feature, create_graph=True)

# Distilling the Knowledge in a Neural Network
def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd

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

def labels_to_prompts(target):
    prompts = ["This is a image of number " + str(int(label)) for label in target]
    return prompts

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

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

class SRT(Distiller):
    """soft relaxation taylor approximation
    """

    def __init__(self, student, teacher, cfg):
        super(SRT, self).__init__(student, teacher)
        self.p = cfg.AT.P

        # vanilla kd setting
        self.temperature = cfg.KD.TEMPERATURE
        
        self.freeze(self.teacher)
        self.kd_temperature = cfg.KD.TEMPERATURE

        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        # add model

        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        # teaacher have been freeze
        logits_teacher, feature_teacher = self.teacher(image)

        b, c, h, w = feature_teacher["feats"][-1].shape # 8 * 256 * 8 * 8
        # text_adaptives = [self.adaptive_layer(text_feature.float()) for text_feature in text_features]
        # text_adaptives = torch.stack(text_adaptives, dim=0)
        # res_t_f = self.transformer(text_adaptives.unsqueeze(-1).permute(0,2,1), feature_teacher["feats"][-1].\
        #                            view(b, c, -1).permute(0,2,1)).permute(0,2,1).view(b, c, h, w)
        # res_t_f = self.transformer(feature_teacher["feats"][-1].view(b, c, -1).permute(0,2,1), text_adaptives.unsqueeze(-1).permute(0,2,1)).permute(0,2,1).view(b, c, h, w)

        # losses
        ce_loss_weight = 1
        loss_ce = ce_loss_weight * F.cross_entropy(logits_student, target)

        # CrossKD
        # kd_logits = self.teacher.fc(nn.AvgPool2d(h)(feature_student["feats"][-1]).reshape(b, -1))\
        # with torch.no_grad():
        kd_logits = self.student.fc(nn.AvgPool2d(h)(feature_teacher["feats"][-1]).reshape(b, -1)).detach()
        kd_loss_weight = 100
        # loss_kd = kd_loss_weight * kd_loss(kd_logits, logits_teacher, self.kd_temperature)
        loss_kd = kd_loss_weight * kd_loss(logits_student, kd_logits, self.kd_temperature)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
