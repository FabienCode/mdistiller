import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes


class FeatureDiffusion(nn.Module):
    def __init__(self, feature_dim, num_steps, model):
        super(FeatureDiffusion, self).__init__()
        self.num_steps = num_steps
        self.feature_dim = feature_dim
        self.model = model  # 假设 model 用于特征的去噪

    def forward(self, original_features):
        enhanced_features = []
        noisy_features = original_features.clone()

        for step in range(self.num_steps):
            # 加噪声
            noise_level = (step + 1) / self.num_steps
            noisy_features = self.add_noise(noisy_features, noise_level)

            # 去噪声并保存增强特征
            denoised_features = self.model(noisy_features)
            enhanced_features.append(denoised_features)

        return enhanced_features

    def add_noise(self, features, noise_level):
        return features + noise_level * torch.randn_like(features)


class MVKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(MVKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.FITNET.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT
        self.hint_layer = cfg.FITNET.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        loss_feat = self.feat_loss_weight * F.mse_loss(
            f_s, feature_teacher["feats"][self.hint_layer]
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
        }
        return logits_student, losses_dict
