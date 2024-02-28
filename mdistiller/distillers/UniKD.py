import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes


class UniKDFitNetKd(Distiller):
    """FitNets: Hints for Thin Deep Nets"""

    def __init__(self, student, teacher, cfg):
        super(UniKDFitNetKd, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.FITNET.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT
        self.hint_layer = 3
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )

        # self.feat2pro_s = featPro(feat_s_shapes[-1][1], feat_s_shapes[-1][1], feat_s_shapes[-1][-1], self.class_num)
        self.feat2pro = featPro(feat_t_shapes[self.hint_layer][1], feat_t_shapes[self.hint_layer][-1], 100)

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters())+ list(self.feat2pro.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        f_s_uni = self.feat2pro(f_s)
        f_t_uni = self.feat2pro(feature_teacher["feats"][self.hint_layer])
        loss_feat = kd_loss(
            f_s_uni, f_t_uni, 4
        )
        loss_kd = kd_loss(
            logits_student, logits_teacher, 4
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_feat": loss_feat,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
    

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class featPro(nn.Module):
    # Review KD version
    def __init__(self, in_channels, size, num_classes):
        super(featPro, self).__init__()
        # self.encoder = nn.Sequential(
        #     # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(in_channels),
        #     # nn.LeakyReLU(inplace=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(latent_dim),
        #     # nn.LeakyReLU(inplace=True),
        #     nn.ReLU(inplace=True),
        # )
        # self.feature_adapt = nn.Sequential(
        #         nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        #         nn.ReLU(),
        #         nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        #         nn.ReLU(),
        #         nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        # )
        # self.layernorm = nn.GroupNorm(num_groups=1, num_channels=in_channels, affine=False)
        # self.avg_pool = nn.AvgPool2d((size, size))
        self.fc_mu = nn.Linear(in_channels, num_classes)
        self.fc_var = nn.Linear(in_channels, num_classes)
        # self.fc_mu = nn.Sequential(
        #     nn.Linear(in_channels, latent_dim),
        #     nn.Linear(latent_dim, num_classes)
        # )

    def encode(self, x):
        b, c, h, w = x.shape
        # x = self.layernorm(self.feature_adapt(x))
        # result = self.encoder(x)
        # result = result.view(result.size(0), -1)
        res_pooled = nn.AvgPool2d((h, w))(x).reshape(b, -1)
        mu = self.fc_mu(res_pooled)
        log_var = self.fc_var(res_pooled)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, log_var = self.encode(x)
        # z = torch.cat((mu, log_var), dim=1)
        # z = mu + log_var
        z = self.reparameterize(mu, log_var)
        return z
