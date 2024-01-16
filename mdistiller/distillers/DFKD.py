import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes

from mdistiller.engine.dfkd_primitives import sub_policies
import random
from mdistiller.engine.dfkd_utils import MixedAugment, QFunc
from torch.autograd import Variable
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
# from mdistiller.distillers import distiller_dict


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class DFKD(Distiller):
    """Differentiable Feature Augmentation for Knowledge Distillation."""

    def __init__(self, student, teacher, cfg):
        super(DFKD, self).__init__(student, teacher)
        self.cfg = cfg
        self.ce_loss_weight = cfg.DFKD.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.DFKD.LOSS.FEAT_WEIGHT
        self.hint_layer = cfg.FITNET.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )

        # DFKD submodule
        self.sub_policies = random.sample(sub_policies, 105)
        self.mix_augment = MixedAugment(sub_policies)
        self.augmenting = True
        # self.augment_parameters = self._initialize_augment_parameters()
        self._initialize_augment_parameters()
        self.temperature = 0.5

        # DFKD config
        self.dfkd_t = cfg.DFKD.TEMPERATURE
        self.kd_loss_weight = cfg.LOSS.KD_WEIGHT

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student, logits_teacher, feature_teacher = self.forward_feat(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        f_t = feature_teacher["feats"][self.hint_layer]
        b, c, h, w = f_t.shape
        if self.augmenting:
            aug_f_t = self.mix_augment.forward(f_t, self.probabilities_b, self.magnitudes, self.ops_weights_b)
        else:
            aug_f_t = f_t
        share_classifier = self.teacher.fc
        with torch.no_grad():
            share_f_t = share_classifier(nn.AvgPool2d(h)(aug_f_t).reshape(b, -1))
            share_f_s = share_classifier(nn.AvgPool2d(h)(f_s).reshape(b, -1))
        loss_kd = self.kd_loss_weight * kd_loss(
            share_f_s, share_f_t, self.temperature
        )
        loss_feat = self.feat_loss_weight * F.mse_loss(f_s, aug_f_t)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_feat": loss_feat,
            "loss_kd": loss_kd,
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
        self.probabilities = Variable(0.5*torch.ones(num_sub_policies, num_ops).cuda(), requires_grad=True)
        self.ops_weights = Variable(1e-3*torch.ones(num_sub_policies).cuda(), requires_grad=True)
        self.q_func = [QFunc(num_sub_policies*(num_ops+1)).cuda()]
        self.magnitudes = Variable(0.5*torch.ones(num_sub_policies, num_ops).cuda(), requires_grad=True)
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
        d_logits_list = d_logits_list +list( d_q_func )
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
        distiller = DFKD(model_student, model_teacher, self.cfg)
        distiller = torch.nn.DataParallel(distiller.cuda())
        return distiller
