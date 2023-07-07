import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


from ._base import Distiller
from ..engine.kd_loss import KDQualityFocalLoss, kd_loss




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
        # s_feat = self.align_scale(feature_student["feats"][-1], feature_teacher["feats"][-1])
        # kd_logits = self.teacher.fc(nn.AvgPool2d(h)(s_feat).reshape(b, -1))
        kd_logits = self.teacher.fc(nn.AvgPool2d(h)(feature_student["feats"][-1]).reshape(b, -1))
        # with torch.no_grad():
        #     kd_logits = self.student.fc(nn.AvgPool2d(h)(feature_teacher["feats"][-1]).reshape(b, -1))
        kd_loss_weight = 1
        # loss_kd = kd_loss_weight * kd_loss(kd_logits, logits_teacher, self.kd_temperature)
        loss_kd = kd_loss_weight * kd_loss(logits_student, kd_logits, self.kd_temperature)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
    
    def align_scale(self, s_feat, t_feat):
        N, C, H, W = s_feat.size()
        # normalize source feature
        s_feat = s_feat.permute(1, 0, 2, 3).reshape(C, -1)
        s_mean = s_feat.mean(dim=-1, keepdim=True)
        s_std = s_feat.std(dim=-1, keepdim=True)
        s_feat = (s_feat - s_mean) / (s_std + 1e-6)

        #
        t_feat = t_feat.permute(1, 0, 2, 3).reshape(C, -1)
        t_mean = t_feat.mean(dim=-1, keepdim=True)
        t_std = t_feat.std(dim=-1, keepdim=True)
        s_feat = s_feat * t_std + t_mean
        return s_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)
