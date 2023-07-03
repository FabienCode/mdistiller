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
        self.ce_loss_weight = cfg.AT.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.AT.LOSS.FEAT_WEIGHT

        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT

        # vanilla kd setting
        self.temperature = cfg.KD.TEMPERATURE
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        
        self.freeze(self.teacher)
        self.teacher_fc = self.teacher.fc
        self.kd_temperature = cfg.KD.TEMPERATURE

        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        # add model
        # self.clip_model, self.preprocess = clip.load("ViT-B/32", jit=False)
        # self.transformer = nn.Transformer(d_model=256, batch_first=True)
        # self.adaptive_layer = nn.Linear(512, 256)

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def forward_train(self, image, target, **kwargs):
        # prompts = labels_to_prompts(target)
        # prompts_text = clip.tokenize(prompts).cuda()
        # with torch.no_grad():
        #     text_features = self.clip_model.encode_text(prompts_text)
        logits_student, feature_student = self.student(image)
        # with torch.no_grad():
        logits_teacher, feature_teacher = self.teacher(image)

        b, c, h, w = feature_teacher["feats"][-1].shape # 8 * 256 * 8 * 8
        # text_adaptives = [self.adaptive_layer(text_feature.float()) for text_feature in text_features]
        # text_adaptives = torch.stack(text_adaptives, dim=0)
        # res_t_f = self.transformer(text_adaptives.unsqueeze(-1).permute(0,2,1), feature_teacher["feats"][-1].\
        #                            view(b, c, -1).permute(0,2,1)).permute(0,2,1).view(b, c, h, w)
        # res_t_f = self.transformer(feature_teacher["feats"][-1].view(b, c, -1).permute(0,2,1), text_adaptives.unsqueeze(-1).permute(0,2,1)).permute(0,2,1).view(b, c, h, w)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # loss_feat = 0.1 * F.mse_loss(res_t_f, feature_student["feats"][-1])
        # loss_feat = self.feat_loss_weight * at_loss(
        #     feature_student["feats"][1:], feature_teacher["feats"][1:], self.p
        # )
        # loss_feat = self.feat_loss_weight * at_loss(
        #     feature_student["feats"], feature_teacher["feats"], self.p
        # )
        # loss_vanilla_kd = self.kd_loss_weight * kd_loss(
        #     logits_student, logits_teacher, self.temperature
        # )
        # teacher_kd_feature = feature_teacher["feats"][-1]
        # student_kd_feature = feature_student["feats"][-1]

        # CrossKD--A
        # kd_student_logits = self.teacher_fc(nn.AvgPool2d(h)(feature_student["feats"][-1]).reshape(b, -1))
        # loss_kd = kd_loss(kd_student_logits, logits_teacher, self.kd_temperature)
        # CrossKD--B
        kd_teacher_logits = self.student.fc(nn.AvgPool2d(h)(feature_teacher["feats"][-1]).reshape(b, -1))
        loss_kd = 0.3 * kd_loss(kd_teacher_logits, logits_student, self.kd_temperature)
        # CrossKD--C
        # kd_student_logits = self.teacher_fc(nn.AvgPool2d(h)(feature_student["feats"][-1]).reshape(b, -1))
        # loss_kd = kd_loss(kd_student_logits, logits_student, self.kd_temperature)
        # CrossKD--D
        # kd_teacher_logits = self.student.fc(nn.AvgPool2d(h)(feature_teacher["feats"][-1]).reshape(b, -1))
        # loss_kd = kd_loss(kd_teacher_logits, logits_teacher, self.kd_temperature)


        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
