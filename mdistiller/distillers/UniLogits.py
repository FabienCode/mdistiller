import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
from mdistiller.engine.kd_loss import mask_logits_loss, dkd_loss
from mdistiller.engine.uni_utils import Feat2ProAttention, ChannelAttentionModule, UNet


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd


def feature_dis_loss(feature_student, feature_teacher, temperature):
    b, c, h, w = feature_student.shape
    mean_s = torch.mean(feature_student.reshape(b, c, -1), dim=1)
    mean_t = torch.mean(feature_teacher.reshape(b, c, -1), dim=1)
    # log_s = F.log_softmax(mean_s / temperature, dim=1)
    # log_t = F.softmax(mean_t / temperature, dim=1)
    # loss = F.kl_div(log_s, log_t, reduction="none").sum(1).mean()
    # loss *= temperature**2
    loss = F.mse_loss(mean_s, mean_t)
    # loss = F.kl_div(F.normalize(mean_s), F.normalize(mean_t), reduction='batchmean')
    return loss


def feature_dkd_dis_loss(feature_student, feature_teacher, target, alpha, beta, temperature):
    b, c, h, w = feature_student.shape
    mean_s = torch.mean(feature_student.reshape(b, c, -1), dim=1)
    mean_t = torch.mean(feature_teacher.reshape(b, c, -1), dim=1)
    mask = torch.zeros_like(mean_s)
    max_indices = torch.argmax(mean_t, dim=-1)
    mask[torch.arange(b), max_indices] = 1
    loss = mask_logits_loss(mean_s, mean_t, target, alpha, beta, temperature, mask.bool())
    return loss


class UniLogitsKD(Distiller):
    """Review KD & DKD logits with UniKD"""

    def __init__(self, student, teacher, cfg):
        super(UniLogitsKD, self).__init__(student, teacher)
        self.temperature = cfg.Uni.TEMPERATURE
        self.ce_loss_weight = cfg.Uni.LOSS.CE_WEIGHT
        self.logits_weight = cfg.Uni.LOSS.LOGITS_WEIGHT
        self.feat_weight = cfg.Uni.LOSS.FEAT_KD_WEIGHT
        self.supp_weight = cfg.Uni.LOSS.SUPP_WEIGHT

        # dkd para
        self.warmup = cfg.Uni.WARMUP
        self.alpha = cfg.Uni.ALPHA
        self.beta = cfg.Uni.BETA
        self.class_num = cfg.Uni.CLASS_NUM
        self.latent_dim = cfg.Uni.LATENT_DIM
        self.supp_t = cfg.Uni.SUPP_T

        self.hint_layer = cfg.Uni.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.Uni.INPUT_SIZE
        )

        # Review KD config
        self.shapes = cfg.Uni.SHAPES
        self.out_shapes = cfg.Uni.OUT_SHAPES
        in_channels = cfg.Uni.IN_CHANNELS
        out_channels = cfg.Uni.OUT_CHANNELS
        self.stu_preact = cfg.Uni.STU_PREACT
        self.max_mid_channel = cfg.Uni.MAX_MID_CHANNEL
        # origin (student) fuse module
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
        # self.feat2pro = featPro(out_channels[0], min(256, out_channels[-1]), self.shapes[-1], self.class_num)

        self.feat2pro_s = featPro(out_channels[0], min(256, out_channels[-1]), self.shapes[-1], self.class_num)
        self.feat2pro_t = featPro(out_channels[0], min(256, out_channels[-1]), self.shapes[-1], self.class_num)

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

    # Test
    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        # get features
        if self.stu_preact:
            x = feature_student["preact_feats"] + [
                feature_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        else:
            x = feature_student["feats"] + [
                feature_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        x = x[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(
            x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)
        features_teacher = feature_teacher["preact_feats"][1:] + [
            feature_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_logits = self.logits_weight * min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        # gitee
        # f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        # f_t = feature_teacher["feats"][self.hint_layer]
        f_s = results[0]
        f_t = features_teacher[0]
        f_s_pro = self.feat2pro_s(f_s)
        f_t_pro = self.feat2pro_t(f_t)
        loss_feat = self.feat_weight * kd_loss(f_s_pro, f_t_pro, self.supp_t)
        # loss_feat = self.feat_weight * F.mse_loss(f_s_pro, f_t_pro)
        loss_supp_feat2pro = self.supp_weight * (
                kd_loss(f_s_pro, logits_student, self.supp_t) + kd_loss(f_t_pro, logits_teacher,
                                                                        self.supp_t))
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_feature": loss_feat,
            "loss_logits": loss_logits,
            "loss_supp_feat2pro": loss_supp_feat2pro
        }
        return logits_student, losses_dict


# class UniLogitsKD(Distiller):
#     """Distilling the Knowledge in a Neural Network"""
#
#     def __init__(self, student, teacher, cfg):
#         super(UniLogitsKD, self).__init__(student, teacher)
#         self.temperature = cfg.Uni.TEMPERATURE
#         self.ce_loss_weight = cfg.Uni.LOSS.CE_WEIGHT
#         self.logits_weight = cfg.Uni.LOSS.LOGITS_WEIGHT
#         self.feat_weight = cfg.Uni.LOSS.FEAT_KD_WEIGHT
#         self.supp_weight = cfg.Uni.LOSS.SUPP_WEIGHT
#
#         # dkd para
#         self.warmup = cfg.Uni.WARMUP
#         self.alpha = cfg.Uni.ALPHA
#         self.beta = cfg.Uni.BETA
#         self.class_num = cfg.Uni.CLASS_NUM
#         self.latent_dim = cfg.Uni.LATENT_DIM
#         self.supp_t = cfg.Uni.SUPP_T
#
#         self.hint_layer = cfg.Uni.HINT_LAYER
#         feat_s_shapes, feat_t_shapes = get_feat_shapes(
#             self.student, self.teacher, cfg.Uni.INPUT_SIZE
#         )
#
#         self.conv_reg = ConvReg(
#             feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
#         )
#         self.feat2pro_s = featPro(feat_t_shapes[self.hint_layer][1], feat_t_shapes[self.hint_layer][2], self.latent_dim, self.class_num)
#         self.feat2pro_t = featPro(feat_t_shapes[self.hint_layer][1], feat_t_shapes[self.hint_layer][2], self.latent_dim, self.class_num)
#         # self.feat2pro = featPro(feat_t_shapes[self.hint_layer][1], feat_t_shapes[self.hint_layer][2], self.latent_dim,
#         #                         self.class_num)
#         # self.feat2pro = feat2Pro(feat_t_shapes[self.hint_layer][1], feat_t_shapes[self.hint_layer][2], 256, 100, self.gmm_num)
#         # self.supp_loss = MGDLoss(100, self.mask_rate)
#         # self.feat2pro_s = Feat2ProAttention(feat_t_shapes[self.hint_layer][1], feat_t_shapes[self.hint_layer][2], 256,
#         #                                     self.class_num)
#         # self.feat2pro_t = Feat2ProAttention(feat_t_shapes[self.hint_layer][1], feat_t_shapes[self.hint_layer][2], 256,
#         #                                     self.class_num)
#
#     def get_learnable_parameters(self):
#         return super().get_learnable_parameters() + list(self.conv_reg.parameters()) + \
#             list(self.feat2pro_s.parameters()) + list(self.feat2pro_t.parameters())
#
#     def get_extra_parameters(self):
#         num_p = 0
#         for p in self.conv_reg.parameters():
#             num_p += p.numel()
#         for p in self.feat2pro_t.parameters():
#             num_p += p.numel()
#         for p in self.feat2pro_s.parameters():
#             num_p += p.numel()
#         # for p in self.supp_loss.parameters():
#         #     num_p += p.numel()
#         return num_p
#
#     # Test
#     def forward_train(self, image, target, **kwargs):
#         logits_student, feature_student = self.student(image)
#         with torch.no_grad():
#             logits_teacher, feature_teacher = self.teacher(image)
#
#         # losses
#         loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
#         loss_logits = self.logits_weight * min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
#             logits_student,
#             logits_teacher,
#             target,
#             self.alpha,
#             self.beta,
#             self.temperature,
#         )
#         # gitee
#         f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
#         f_t = feature_teacher["feats"][self.hint_layer]
#         f_s_pro = self.feat2pro_s(f_s)
#         f_t_pro = self.feat2pro_t(f_t)
#         loss_feat = self.feat_weight * kd_loss(f_s_pro, f_t_pro, self.supp_t)
#         # loss_feat = self.feat_weight * F.mse_loss(f_s_pro, f_t_pro)
#         loss_supp_feat2pro = self.supp_weight * (
#                 kd_loss(f_s_pro, logits_student, self.supp_t) + kd_loss(f_t_pro, logits_teacher,
#                                                                         self.supp_t))
#         losses_dict = {
#             "loss_ce": loss_ce,
#             "loss_feature": loss_feat,
#             "loss_logits": loss_logits,
#             "loss_supp_feat2pro": loss_supp_feat2pro
#         }
#         return logits_student, losses_dict


# class featPro(nn.Module):
#     def __init__(self, in_channels, size, latent_dim, num_classes):
#         super(featPro, self).__init__()
#         # self.encoder = ChannelAttentionModule(num_classes, 3, size, size)
#         if in_channels == 2 * latent_dim:
#             self.conv = nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=2, padding=1)
#         elif in_channels * 2 == latent_dim:
#             self.conv = nn.ConvTranspose2d(in_channels, latent_dim, kernel_size=4, stride=2, padding=1)
#         elif in_channels >= latent_dim:
#             self.conv = nn.Conv2d(in_channels, latent_dim, kernel_size=1)
#         else:
#             self.conv = nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1)
#         self.encoder = nn.Sequential(
#             self.conv,
#             nn.ReLU(),
#             # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
#             nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
#             # nn.BatchNorm2d(in_channels),
#             # nn.LeakyReLU(inplace=True),
#             # nn.LayerNorm([in_channels, size, size]),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
#             # nn.LayerNorm([latent_dim, size, size]),
#             # nn.BatchNorm2d(latent_dim),
#             # nn.LeakyReLU(inplace=True),
#             nn.ReLU(inplace=True),
#         )
#         # self.encoder = UNet(in_channels, num_classes, False)
#         self.avg_pool = nn.AvgPool2d((size, size))
#         # self.fc_mu = nn.Linear(latent_dim * size * size, latent_dim)
#         # self.fc_var = nn.Linear(latent_dim * size * size, latent_dim)
#         self.fc_mu = nn.Linear(latent_dim, num_classes)
#         self.fc_var = nn.Linear(latent_dim, num_classes)
#         # self.conv_mu = nn.Sequential(
#         #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
#         #     nn.ReLU(),
#         #     nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1),
#         #     nn.ReLU()
#         # )
#         # self.conv_var = nn.Sequential(
#         #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
#         #     nn.ReLU(),
#         #     nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1),
#         #     nn.ReLU()
#         # )
#
#     def encode(self, x):
#         # res_mu = self.conv_mu(x)
#         # res_var = self.conv_var(x)
#         # res_mu = self.encoder(res_mu)
#         # res_var = self.encoder(res_var)
#         # mu, var = self.encoder(x)
#         # mu = self.avg_pool(mu).view(mu.size(0), -1)
#         # var = self.avg_pool(var).view(var.size(0), -1)
#         # return mu, var
#         result = self.encoder(x)
#         # result = result.view(result.size(0), -1)
#         res_pooled = self.avg_pool(result).reshape(result.size(0), -1)
#         mu = self.fc_mu(res_pooled)
#         log_var = self.fc_var(res_pooled)
#         return mu, log_var
#
#     def reparameterize(self, mu, log_var):
#         std = torch.exp(log_var / 2)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mu)
#
#     def forward(self, x):
#         mu, log_var = self.encode(x)
#         # z = torch.cat((mu, log_var), dim=1)
#         # z = mu + log_var
#         z = self.reparameterize(mu, log_var)
#         return z


# class featPro(nn.Module):
#     # normal version
#     def __init__(self, in_channels, size, latent_dim, num_classes):
#         super(featPro, self).__init__()
#         self.encoder = nn.Sequential(
#             # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
#             # nn.BatchNorm2d(in_channels),
#             # nn.LeakyReLU(inplace=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1),
#             # nn.BatchNorm2d(latent_dim),
#             # nn.LeakyReLU(inplace=True),
#             nn.ReLU(inplace=True),
#         )
#         self.avg_pool = nn.AvgPool2d((size, size))
#         # self.fc_mu = nn.Linear(latent_dim * size * size, latent_dim)
#         # self.fc_var = nn.Linear(latent_dim * size * size, latent_dim)
#         self.fc_mu = nn.Linear(latent_dim, num_classes)
#         self.fc_var = nn.Linear(latent_dim, num_classes)
#
#     def encode(self, x):
#         result = self.encoder(x)
#         # result = result.view(result.size(0), -1)
#         res_pooled = self.avg_pool(result).reshape(result.size(0), -1)
#         mu = self.fc_mu(res_pooled)
#         log_var = self.fc_var(res_pooled)
#         return mu, log_var
#
#     def reparameterize(self, mu, log_var):
#         std = torch.exp(log_var / 2)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mu)
#
#     def forward(self, x):
#         mu, log_var = self.encode(x)
#         # z = torch.cat((mu, log_var), dim=1)
#         # z = mu + log_var
#         z = self.reparameterize(mu, log_var)
#         return z


class featPro(nn.Module):
    # Review KD version
    def __init__(self, in_channels, latent_dim, size, num_classes):
        super(featPro, self).__init__()
        # self.encoder = nn.Sequential(
        #     # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        #     # nn.BatchNorm2d(in_channels),
        #     # nn.LeakyReLU(inplace=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1),
        #     # nn.BatchNorm2d(latent_dim),
        #     # nn.LeakyReLU(inplace=True),
        #     nn.ReLU(inplace=True),
        # )
        self.avg_pool = nn.AvgPool2d((size, size))
        self.fc_mu = nn.Linear(latent_dim, num_classes)
        self.fc_var = nn.Linear(latent_dim, num_classes)
        # self.fc_mu = nn.Sequential(
        #     nn.Linear(in_channels, latent_dim),
        #     nn.Linear(latent_dim, num_classes)
        # )

    def encode(self, x):
        # result = self.encoder(x)
        # result = result.view(result.size(0), -1)
        res_pooled = self.avg_pool(x).reshape(x.size(0), -1)
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

class GaussianMixtureLayer(nn.Module):
    def __init__(self, in_channels, size, latent_dim, num_classes, num_gaussians):
        super(GaussianMixtureLayer, self).__init__()
        self.in_channels = in_channels
        self.size = size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_gaussians = num_gaussians

        # Difine linear layers to predict the parameters of the Gaussian mixture
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AvgPool2d((size, size))
        self.mu_layer = nn.Linear(latent_dim, num_classes * num_gaussians)
        self.sigma_layer = nn.Linear(latent_dim, num_classes * num_gaussians)
        self.pi_layer = nn.Linear(latent_dim, num_gaussians)

    def forward(self, x):
        res = self.encoder(x)
        res_pooled = self.avg_pool(res).reshape(res.size(0), -1)
        mu = self.mu_layer(res_pooled).view(-1, self.num_gaussians, self.num_classes)
        sigma = F.softplus(self.sigma_layer(res_pooled)).view(-1, self.num_gaussians, self.num_classes)
        pi = F.softmax(self.pi_layer(res_pooled), dim=1)  # Mixing coefficients
        return pi, mu, sigma


class MixtureOfGaussians(nn.Module):
    def __init__(self, n_components=2):
        super(MixtureOfGaussians, self).__init__()
        self.n_components = n_components

    # Gumbel-Softmax trick for differentiable sampling
    def gumber_softmax_sample(self, logits, temperature):
        gumbel_noise = -torch.empty_like(logits).exponential_().log()
        sampled_logtis = (logits + gumbel_noise) / temperature
        return F.softmax(sampled_logtis, dim=-1)

    # Reparameterization trick for Gaussian sampling
    def reparameterize(self, mean, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, pi, mu, sigma, temperature=1.0):
        # Convert variance to log variance
        log_var = 2 * sigma.log()

        # Use Gumbel-Softmax to get the continuous approximation of the component weight
        weights = self.gumber_softmax_sample(pi, temperature)

        # Sample from each Gaussian
        samples = [self.reparameterize(mu[:, i], log_var[:, i]) for i in range(self.n_components)]

        # Weighted combination of Gaussian samples
        final_sample = sum([weights[:, i].unsqueeze(1) * samples[i] for i in range(self.n_components)])

        return final_sample


# Test auto update
class feat2Pro(nn.Module):
    def __init__(self, in_channels, size, latent_dim, num_classes, n_components):
        super(feat2Pro, self).__init__()
        self.gaussian_mixture_layer = GaussianMixtureLayer(in_channels, size, latent_dim, num_classes, n_components)
        self.mixture_of_gaussians = MixtureOfGaussians(n_components)

    def forward(self, x, temperature=1.0):
        pi, mu, sigma = self.gaussian_mixture_layer(x)
        final_sample = self.mixture_of_gaussians(pi, mu, sigma, temperature)
        return final_sample


class MGDLoss(nn.Module):
    def __init__(self, channels, alpha_mgd, lambda_mgd=0.5):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )

    def get_dis_loss(self, s, t):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C = s.shape
        device = s.device
        mat = torch.rand((N, C)).to(device)
        mat = torch.where(mat < self.lambda_mgd, 0, 1).to(device)

        masked_feat = torch.mul(s, mat)
        new_feat = self.generation(masked_feat)
        dis_loss = loss_mse(new_feat, t) / N
        return dis_loss

    def forward(self, s, t):
        # assert s.shape[-2:] == t.shape[-2:]
        loss = self.get_dis_loss(s, t) * self.alpha_mgd
        return loss


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            # nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            # nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        # output
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x
#test pycharm