import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


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

def layers_rkd_loss(f_s, f_t, squared=False, eps=1e-12, distance_weight=25, angle_weight=50):
    return sum([rkd_loss(s, t, squared, eps, distance_weight, angle_weight) for s, t in zip(f_s, f_t)])


class RKD(Distiller):
    """Relational Knowledge Disitllation, CVPR2019"""

    def __init__(self, student, teacher, cfg):
        super(RKD, self).__init__(student, teacher)
        self.distance_weight = cfg.RKD.DISTANCE_WEIGHT
        self.angle_weight = cfg.RKD.ANGLE_WEIGHT
        self.ce_loss_weight = cfg.RKD.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.RKD.LOSS.FEAT_WEIGHT
        self.eps = cfg.RKD.PDIST.EPSILON
        self.squared = cfg.RKD.PDIST.SQUARED

        # self.logits_avg = nn.Sequential(
        #     nn.AvgPool2d(32),
        #     nn.AvgPool2d(16),
        #     nn.AvgPool2d(8)
        # )

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)

        bs = feature_teacher["feats"][0].shape[0]
        channels = []
        for i in range(len(feature_student["feats"])):
            channels.append(feature_student["feats"][i].shape[1])
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # kd_pooled_student = []
        # for i in range(len(feature_student["feats"][1:-1])):
        #     kd_pooled_student.append(self.logits_avg[i](feature_student["feats"][i+1]).reshape(bs, -1))
        # kd_pooled_student.append(feature_student["pooled_feat"])
        # kd_pooled_teacher = []
        # for i in range(len(feature_teacher["feats"][1:-1])):
        #     kd_pooled_teacher.append(self.logits_avg[i](feature_teacher["feats"][i+1]).reshape(bs, -1))
        # kd_pooled_teacher.append(feature_teacher["pooled_feat"])
        # loss_rkd = self.feat_loss_weight * layers_rkd_loss(
        #     kd_pooled_student,
        #     kd_pooled_teacher,
        #     self.squared,
        #     self.eps,
        #     self.distance_weight,
        #     self.angle_weight,
        # )
        loss_rkd = self.feat_loss_weight * rkd_loss(
            feature_student["pooled_feat"],
            feature_teacher["pooled_feat"],
            self.squared,
            self.eps,
            self.distance_weight,
            self.angle_weight,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_rkd,
        }
        return logits_student, losses_dict
