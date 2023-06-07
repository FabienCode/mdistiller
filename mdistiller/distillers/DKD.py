import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


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

def layers_dkd(logits_students, logits_teachers, target, alpha, beta, temperature):
    return sum([dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature) for logits_student, logits_teacher in zip(logits_students, logits_teachers)])

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


class DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

        # layer logits modules
        num_classes = student.fc.out_features
        # self.logits_fc = nn.Sequential(
        #     nn.Linear(self.student.layer1[0].conv2.out_channels, num_classes),
        #     nn.Linear(self.student.layer2[0].conv2.out_channels, num_classes),
        #     nn.Linear(self.student.layer3[0].conv2.out_channels, num_classes)
        # )
        self.logits_avg = nn.Sequential(
            nn.AvgPool2d(32),
            nn.AvgPool2d(16),
            nn.AvgPool2d(8)
        )

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        bs = feature_teacher["feats"][0].shape[0]
        channels = []
        for i in range(len(feature_student["feats"])):
            channels.append(feature_student["feats"][i].shape[1])

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # KD loss
        kd_logits_student = []
        for i in range(len(feature_student["feats"][1:-1])):
            # kd_logits_student.append(self.logits_fc[i](self.logits_avg[i](feature_student["feats"][i+1])\
            #                                             .reshape(feature_student["feats"][i+1].shape[0], -1)))
            with torch.no_grad():
                tmp_fc = self.student.fc
                kd_logits_student.append(tmp_fc(self.logits_avg[i](feature_student["feats"][i+1]).repeat(1, int(channels[-1]/channels[i+1]), 1, 1)\
                                                .reshape(bs, -1)))
        kd_logits_student.append(logits_student)
        kd_logits_teacher = []
        for i in range(len(feature_teacher["feats"][1:-1])):
            with torch.no_grad():
                tmp_fc = self.teacher.fc
                kd_logits_teacher.append(tmp_fc(self.logits_avg[i](feature_teacher["feats"][i+1]).repeat(1, int(channels[-1]/channels[i+1]), 1, 1)\
                                                .reshape(bs, -1)))
        kd_logits_teacher.append(logits_teacher)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * layers_dkd(
            kd_logits_student,
            kd_logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        # loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
        #     logits_student,
        #     logits_teacher,
        #     target,
        #     self.alpha,
        #     self.beta,
        #     self.temperature,
        # )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
