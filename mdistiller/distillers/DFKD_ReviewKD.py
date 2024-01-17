import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from mdistiller.engine.dfkd_primitives import sub_policies
import random
from mdistiller.engine.dfkd_utils import MixedAugment, QFunc
from torch.autograd import Variable
from mdistiller.models import cifar_model_dict, imagenet_model_dict

from ._base import Distiller


def hcl_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


class DFKDReviewKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(DFKDReviewKD, self).__init__(student, teacher)
        self.shapes = cfg.REVIEWKD.SHAPES
        self.out_shapes = cfg.REVIEWKD.OUT_SHAPES
        in_channels = cfg.REVIEWKD.IN_CHANNELS
        out_channels = cfg.REVIEWKD.OUT_CHANNELS
        self.ce_loss_weight = cfg.REVIEWKD.DFKD_CE_WEIGHT
        self.reviewkd_loss_weight = cfg.REVIEWKD.DFKD_REVIEWKD_WEIGHT
        self.kd_loss_weight = cfg.REVIEWKD.DFKD_KD_WEIGHT
        self.warmup_epochs = cfg.REVIEWKD.WARMUP_EPOCHS
        self.stu_preact = cfg.REVIEWKD.STU_PREACT
        self.max_mid_channel = cfg.REVIEWKD.MAX_MID_CHANNEL

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

        # DFKD submodule
        self.sub_policies = random.sample(sub_policies, 105)
        self.mix_augment = MixedAugment(sub_policies)
        self.augmenting = True
        # self.augment_parameters = self._initialize_augment_parameters()
        self._initialize_augment_parameters()
        self.temperature = 0.5
        self.cfg = cfg

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.abfs.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.abfs.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        # logits_student, features_student = self.student(image)
        # with torch.no_grad():
        #     logits_teacher, features_teacher = self.teacher(image)
        logits_student, features_student, logits_teacher, features_teacher = self.forward_feat(image)

        # get features
        if self.stu_preact:
            x = features_student["preact_feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        else:
            x = features_student["feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
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
        features_teacher = features_teacher["preact_feats"][1:] + [
            features_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]
        if self.augmenting:
            features_teacher_aug = []
            for feature in features_teacher:
                feature_aug = self.mix_augment(feature, self.probabilities_b, self.magnitudes, self.ops_weights_b)
                features_teacher_aug.append(feature_aug)
        else:
            features_teacher_aug = features_teacher

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_reviewkd = (
            self.reviewkd_loss_weight
            * min(kwargs["epoch"] / self.warmup_epochs, 1.0)
            * hcl_loss(results, features_teacher_aug)
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_reviewkd,
        }
        return logits_student, losses_dict

    def forward_feat(self, image):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)
        return logits_student, feature_student, logits_teacher, feature_teacher

    def _initialize_augment_parameters(self):
        num_sub_policies = len(self.sub_policies)
        num_ops = len(self.sub_policies[0])
        self.probabilities = Variable(0.5 * torch.ones(num_sub_policies, num_ops).cuda(), requires_grad=True)
        self.ops_weights = Variable(1e-3 * torch.ones(num_sub_policies).cuda(), requires_grad=True)
        self.q_func = [QFunc(num_sub_policies * (num_ops + 1)).cuda()]
        self.magnitudes = Variable(0.5 * torch.ones(num_sub_policies, num_ops).cuda(), requires_grad=True)
        # self.ops_weights = Variable(1.0/num_sub_policies*torch.ones(num_sub_policies).cuda(), requires_grad=True)
        # self.ops_weights = Variable(1e-3 * torch.randn(num_sub_policies).cuda(), requires_grad=True)

        self._augment_parameters = [
            self.probabilities,
            self.ops_weights,
            self.magnitudes,
        ]
        self._augment_parameters += [*self.q_func[0].parameters()]

    def augment_parameters(self):
        return self._augment_parameters

    def sample(self):
        # probabilities_dist = torch.distributions.RelaxedBernoulli(
        #     self.temperature, self.probabilities)
        # sample_probabilities = probabilities_dist.rsample()
        # sample_probabilities = sample_probabilities.clamp(0.0, 1.0)
        # self.sample_probabilities = sample_probabilities
        # self.sample_probabilities_index = sample_probabilities >= 0.5
        # self.sample_probabilities = \
        #     self.sample_probabilities_index.float() - sample_probabilities.detach() + sample_probabilities
        EPS = 1e-6
        num_sub_policies = len(self.sub_policies)
        num_ops = len(self.sub_policies[0])
        probabilities_logits = torch.log(self.probabilities.clamp(0.0 + EPS, 1.0 - EPS)) - torch.log1p(
            -self.probabilities.clamp(0.0 + EPS, 1.0 - EPS))
        probabilities_u = torch.rand(num_sub_policies, num_ops).cuda()
        probabilities_v = torch.rand(num_sub_policies, num_ops).cuda()
        probabilities_u = probabilities_u.clamp(EPS, 1.0)
        probabilities_v = probabilities_v.clamp(EPS, 1.0)
        probabilities_z = probabilities_logits + torch.log(probabilities_u) - torch.log1p(-probabilities_u)
        probabilities_b = probabilities_z.gt(0.0).type_as(probabilities_z)

        def _get_probabilities_z_tilde(logits, b, v):
            theta = torch.sigmoid(logits)
            v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
            z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
            return z_tilde

        probabilities_z_tilde = _get_probabilities_z_tilde(probabilities_logits, probabilities_b, probabilities_v)
        # probabilities_logits[torch.isnan(probabilities_logits)] = 0.
        # probabilities_b[torch.isnan(probabilities_b)] = 0.
        probabilities_logits = torch.where(torch.isnan(probabilities_logits), torch.zeros_like(probabilities_logits), probabilities_logits)
        probabilities_b = torch.where(torch.isnan(probabilities_b), torch.zeros_like(probabilities_b), probabilities_b)
        self.probabilities_logits = probabilities_logits
        self.probabilities_b = probabilities_b
        self.probabilities_sig_z = torch.sigmoid(probabilities_z / self.temperature)
        self.probabilities_sig_z_tilde = torch.sigmoid(probabilities_z_tilde / self.temperature)

        ops_weights_p = torch.nn.functional.softmax(self.ops_weights, dim=-1)
        ops_weights_logits = torch.log(ops_weights_p)
        ops_weights_u = torch.rand(num_sub_policies).cuda()
        ops_weights_v = torch.rand(num_sub_policies).cuda()
        ops_weights_u = ops_weights_u.clamp(EPS, 1.0)
        ops_weights_v = ops_weights_v.clamp(EPS, 1.0)
        ops_weights_z = ops_weights_logits - torch.log(-torch.log(ops_weights_u))
        ops_weights_b = torch.argmax(ops_weights_z, dim=-1)

        def _get_ops_weights_z_tilde(logits, b, v):
            theta = torch.exp(logits)
            z_tilde = -torch.log(-torch.log(v) / theta - torch.log(v[b]))
            z_tilde = z_tilde.scatter(dim=-1, index=b, src=-torch.log(-torch.log(v[b])))
            # v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
            # z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
            return z_tilde

        ops_weights_z_tilde = _get_ops_weights_z_tilde(ops_weights_logits, ops_weights_b, ops_weights_v)
        self.ops_weights_logits = ops_weights_logits
        self.ops_weights_b = ops_weights_b
        self.ops_weights_softmax_z = torch.nn.functional.softmax(ops_weights_z / self.temperature, dim=-1)
        self.ops_weights_softmax_z_tilde = torch.nn.functional.softmax(ops_weights_z_tilde / self.temperature, dim=-1)

    def set_augmenting(self, value):
        assert value in [False, True]
        self.augmenting = value

    def relax(self, f_b):
        f_z = self.q_func[0](self.probabilities_sig_z, self.ops_weights_softmax_z)
        f_z_tilde = self.q_func[0](self.probabilities_sig_z_tilde, self.ops_weights_softmax_z_tilde)
        probabilities_log_prob = torch.distributions.Bernoulli(logits=self.probabilities_logits).log_prob(self.probabilities_b)
        ops_weights_log_prob = torch.distributions.Categorical(logits=self.ops_weights_logits).log_prob(self.ops_weights_b)
        log_prob = probabilities_log_prob + ops_weights_log_prob
        d_log_prob_list = torch.autograd.grad(
            [log_prob], [self.probabilities, self.ops_weights], grad_outputs=torch.ones_like(log_prob),
            retain_graph=True)
        d_f_z_list = torch.autograd.grad(
            [f_z], [self.probabilities, self.ops_weights], grad_outputs=torch.ones_like(f_z),
            create_graph=True, retain_graph=True)
        d_f_z_tilde_list = torch.autograd.grad(
            [f_z_tilde], [self.probabilities, self.ops_weights], grad_outputs=torch.ones_like(f_z_tilde),
            create_graph=True, retain_graph=True)
        diff = f_b - f_z_tilde
        d_logits_list = [diff * d_log_prob + d_f_z - d_f_z_tilde for
                         (d_log_prob, d_f_z, d_f_z_tilde) in zip(d_log_prob_list, d_f_z_list, d_f_z_tilde_list)]
        # print([d_logits.shape for d_logits in d_logits_list])
        var_loss_list = ([d_logits ** 2 for d_logits in d_logits_list])
        # print([var_loss.shape for var_loss in var_loss_list])
        var_loss = torch.cat([var_loss_list[0], var_loss_list[1].unsqueeze(dim=-1)], dim=-1).mean()
        # var_loss.backward()
        d_q_func = torch.autograd.grad(var_loss, self.q_func[0].parameters(), retain_graph=True)
        d_logits_list = d_logits_list + list(d_q_func)
        return [d_logits.detach() for d_logits in d_logits_list]

    def new(self):
        if self.cfg.DATASET.TYPE == "imagenet":
            num_classes = 1000
            model_teacher = imagenet_model_dict[self.cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[self.cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            num_classes = 100
            net, pretrain_model_path = cifar_model_dict[self.cfg.DISTILLER.TEACHER]
            assert (
                    pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(self.cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            # model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[self.cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = DFKDReviewKD(model_student, model_teacher, self.cfg)
        distiller = torch.nn.DataParallel(distiller.cuda())
        return distiller


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
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
