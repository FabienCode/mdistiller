from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
from .UniLogits import ABF, featPro

def kd_loss(logits_student, logits_teacher, temperature, reduce=True):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd


def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_conf(x, y, lam, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = lam.reshape(-1,1,1,1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class UniMLKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(UniMLKD, self).__init__(student, teacher)
        self.temperature = cfg.Uni.TEMPERATURE
        self.ce_loss_weight = cfg.Uni.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.Uni.LOSS.FEAT_KD_WEIGHT
        self.feat_weight = cfg.Uni.LOSS.FEAT_KD_WEIGHT
        self.supp_weight = cfg.Uni.LOSS.SUPP_WEIGHT

        # feat2pro config
        self.shapes = cfg.Uni.SHAPES
        self.out_shapes = cfg.Uni.OUT_SHAPES
        in_channels = cfg.Uni.IN_CHANNELS
        out_channels = cfg.Uni.OUT_CHANNELS
        self.class_num = cfg.Uni.CLASS_NUM
        self.stu_preact = cfg.Uni.STU_PREACT
        abfs = nn.ModuleList()
        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                )
            )
        self.abfs = abfs[::-1]
        self.feat2pro_s = featPro(out_channels[0], 512, self.shapes[-1], self.class_num)
        self.feat2pro_t = featPro(out_channels[0], 512, self.shapes[-1], self.class_num)

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, feature_student_weak = self.student(image_weak)
        logits_student_strong, feature_student_strong = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, feature_teacher_weak = self.teacher(image_weak)
            logits_teacher_strong, feature_teacher_strong = self.teacher(image_strong)

        batch_size, class_num = logits_student_strong.shape

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
        loss_kd_weak = self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
            # reduce=False
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
            # reduce=False
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
            # reduce=False
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
            # reduce=False
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
            # reduce=False
        ) * mask).mean())

        loss_kd_strong = self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        )

        loss_cc_weak = self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
            # reduce=False
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * class_conf_mask).mean())
        loss_cc_strong = self.kd_loss_weight * cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
        ) + self.kd_loss_weight * cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
        ) + self.kd_loss_weight * cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
        ) + self.kd_loss_weight * cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) + self.kd_loss_weight * cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        )
        loss_bc_weak = self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * mask).mean())
        loss_bc_strong = self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            6.0,
        ) * mask).mean())
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd_weak + loss_kd_strong,
            "loss_cc": loss_cc_weak + loss_cc_strong,
            "loss_bc": loss_bc_weak + loss_bc_strong
        }

        # get features
        if self.stu_preact:
            x_weak = feature_student_weak["preact_feats"] + [
                feature_student_weak["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
            x_strong = feature_student_strong["preact_feats"] + [
                feature_student_strong["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        else:
            x_weak = feature_student_weak["feats"] + [
                feature_student_weak["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
            x_strong = feature_student_strong["feats"] + [
                feature_student_strong["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        x_weak = x_weak[::-1]
        x_strong = x_strong[::-1]
        results_weak = []
        out_features_weak, res_features_weak = self.abfs[0](x_weak[0], out_shape=self.out_shapes[0])
        results_weak.append(out_features_weak)
        for features, abf, shape, out_shape in zip(
                x_weak[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            out_features, res_features = abf(features, res_features_weak, shape, out_shape)
            results_weak.insert(0, out_features)
        features_teacher_weaks = feature_teacher_weak["preact_feats"][1:] + [
            feature_teacher_weak["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]
        f_s_weak = results_weak[0]
        f_t_weak = features_teacher_weaks[0]
        f_s_pro_weak = self.feat2pro_s(f_s_weak)
        f_t_pro_weak = self.feat2pro_t(f_t_weak)
        loss_feat_weak = self.feat_weight * kd_loss(f_s_pro_weak, f_t_pro_weak, self.temperature)
        results_strong = []
        out_features_strong, res_features_strong = self.abfs[0](x_strong[0], out_shape=self.out_shapes[0])
        results_strong.append(out_features_strong)
        for features, abf, shape, out_shape in zip(
                x_strong[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            out_features, res_features = abf(features, res_features_strong, shape, out_shape)
            results_strong.insert(0, out_features)
        feature_teacher_strongs = feature_teacher_strong["preact_feats"][1:] + [
            feature_teacher_strong["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]
        f_s_strong = results_strong[0]
        f_t_strong = feature_teacher_strongs[0]
        f_s_pro_strong = self.feat2pro_s(f_s_strong)
        f_t_pro_strong = self.feat2pro_t(f_t_strong)
        loss_feat_strong = self.feat_weight * kd_loss(f_s_pro_strong, f_t_pro_strong, self.temperature)
        loss_feat = loss_feat_weak + loss_feat_strong
        # loss_supp_feat2pro = self.supp_weight * (
        #         kd_loss(f_s_pro, logits_student, self.supp_t) + kd_loss(f_t_pro, logits_teacher,
        #                                                                 self.supp_t))
        loss_supp_feat2pro = self.supp_weight * (
            kd_loss(f_s_pro_weak, logits_student_weak, self.temperature) + kd_loss(f_t_pro_weak, logits_teacher_weak, self.temperature)
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd_weak + loss_kd_strong,
            "loss_cc": loss_cc_weak,
            "loss_bc": loss_bc_weak + loss_bc_strong,
            "loss_feat": loss_feat,
            "loss_supp_feat2pro": loss_supp_feat2pro,
        }
        return logits_student_weak, losses_dict

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.abfs.parameters()) + \
            list(self.feat2pro_s.parameters()) + list(self.feat2pro_t.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.abfs.parameters():
            num_p += p.numel()
        for p in self.feat2pro_s.parameters():
            num_p += p.numel()
        for p in self.feat2pro_t.parameters():
            num_p += p.numel()
        # for p in self.supp_loss.parameters():
        #     num_p += p.numel()
        return num_p


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss