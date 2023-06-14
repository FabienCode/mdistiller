import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def layer_kd_loss(logits_students, logits_teachers, temperature):
    return sum([kd_loss(logits_student, logits_teacher, temperature) \
                for logits_student, logits_teacher in zip(logits_students, logits_teachers)])


class CLD(Distiller):
    """Chain-of-Logits knowledge distillation via masked layers knowledge"""

    def __init__(self, student, teacher, cfg):
        super(CLD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.num_classes = student.fc.out_features
        self.logits_avg = nn.Sequential(
            nn.AvgPool2d(32),
            nn.AvgPool2d(16),
            nn.AvgPool2d(8)
        )
        self.logits_bn = nn.BatchNorm2d(256)
        

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        # layers logits
        # student logtis
        bs = feature_teacher["feats"][0].shape[0]
        channels = [feat.shape[1] for feat in feature_student["feats"]]

        # add final avg pool feature
        # mask version
        # pooled_teacher_features = [(self.logits_avg[i](feature_teacher["feats"][i+1]).repeat(1, int(channels[-1]/channels[i+1]), 1, 1)\
        #                                         .reshape(bs, -1)) for i in range(1,3)]
        # pooled_teacher_features.append(feature_teacher["pooled_feat"])
        # for i in range(len(pooled_teacher_features)):
        #     pooled_teacher_features[i][torch.rand(8,100)>0.5] = 0
        # pooled_student_features = [(self.logits_avg[i](feature_student["feats"][i+1]).repeat(1, int(channels[-1]/channels[i+1]), 1, 1)\
        #                                         .reshape(bs, -1)) for i in range(1,3)]
        # pooled_student_features.append(feature_student["pooled_feat"])
        # for i in range(len(pooled_student_features)):
        #     pooled_student_features[i][torch.rand(8,100)>0.5] = 0
        # direct add and BN version
        pooled_teacher_features = [self.logits_bn(self.logits_avg[i]((feature_teacher["feats"][i+1]).repeat(1, int(channels[-1]/channels[i+1]), 1, 1))\
                                                 + feature_teacher["pooled_feat"].reshape(bs, -1, 1, 1)).reshape(bs, -1) for i in range(1,3)]
        pooled_teacher_features.append(feature_teacher["pooled_feat"])
        pooled_student_features = [self.logits_bn(self.logits_avg[i]((feature_student["feats"][i+1]).repeat(1, int(channels[-1]/channels[i+1]), 1, 1))\
                                                    + feature_student["pooled_feat"].reshape(bs, -1, 1, 1)).reshape(bs, -1) for i in range(1,3)]
        pooled_student_features.append(feature_student["pooled_feat"])

        with torch.no_grad():
            tmp_fc = self.student.fc
            logits_students = [tmp_fc(feat) for feat in pooled_student_features]
            logits_teachers = [tmp_fc(feat) for feat in pooled_teacher_features]
            for i in range(len(logits_students)-1):
                mask = torch.rand(bs, self.num_classes) > 0.5
                logits_students[i][mask] = 0
                logits_teachers[i][mask] = 0
        
        # logtis_students = []
        # for i in range(len(feature_student["feats"][1:-1])):
        #     with torch.no_grad():
        #         tmp_s_fc = self.student.fc
        #         logtis_students.append(tmp_s_fc(self.logits_avg[i](feature_student["feats"][i+1]).repeat(1, int(channels[-1]/channels[i+1]), 1, 1)\
        #                                         .reshape(bs, -1)))
        # logtis_students.append(logits_student)
        # # teacher logits
        # logtis_teachers = []
        # for i in range(len(feature_teacher["feats"][1:-1])):
        #     with torch.no_grad():
        #         tmp_t_fc = self.student.fc
        #         logtis_teachers.append(tmp_t_fc(self.logits_avg[i](feature_teacher["feats"][i+1]).repeat(1, int(channels[-1]/channels[i+1]), 1, 1)\
        #                                         .reshape(bs, -1)))
        # logtis_teachers.append(logits_teacher)
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # loss_kd = self.kd_loss_weight * kd_loss(
        #     logits_student, logits_teacher, self.temperature
        # )
        loss_kd_layers = self.kd_loss_weight * layer_kd_loss(logits_students, logits_teachers, self.temperature)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd_layers,
        }
        return logits_student, losses_dict
