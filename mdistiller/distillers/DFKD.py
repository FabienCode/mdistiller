import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes

from mdistiller.engine.dfkd_primitives import sub_policies
import random
from mdistiller.engine.dfkd_utils import MixedAugment, QFunc
from torch.autograd import Variable


class DFKD(Distiller):
    """Differentiable Feature Augmentaion for Knowledge Distillation."""

    def __init__(self, student, teacher, cfg):
        super(DFKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.FITNET.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT
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
        self._initialize_augment_parameters()

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
        f_t = feature_teacher["feats"][self.hint_layer]
        loss_feat = self.feat_loss_weight * F.mse_loss(f_s, f_t)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
        }
        return logits_student, losses_dict

    def _initialize_augment_parameters(self):
        num_sub_policies = len(self.sub_policies)
        num_ops = len(self.sub_policies[0])
        self.probabilities = Variable(0.5*torch.ones(num_sub_policies, num_ops).cuda(), requires_grad=True)
        self.magnitudes = Variable(0.5*torch.ones(num_sub_policies, num_ops).cuda(), requires_grad=True)
        self.ops_weights = Variable(1e-3*torch.ones(num_sub_policies).cuda(), requires_grad=True)
        # self.ops_weights = Variable(1.0/num_sub_policies*torch.ones(num_sub_policies).cuda(), requires_grad=True)
        # self.ops_weights = Variable(1e-3 * torch.randn(num_sub_policies).cuda(), requires_grad=True)

        self._augment_parameters = [
            self.probabilities,
            self.magnitudes,
            self.ops_weights
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
        probabilities_logits = torch.log(self.probabilities.clamp(0.0+EPS, 1.0-EPS)) - torch.log1p(-self.probabilities.clamp(0.0+EPS, 1.0-EPS))
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
        self.probabilities_logits = probabilities_logits
        self.probabilities_b = probabilities_b
        self.probabilities_sig_z = torch.sigmoid(probabilities_z/self.temperature)
        self.probabilities_sig_z_tilde = torch.sigmoid(probabilities_z_tilde/self.temperature)

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
            z_tilde = -torch.log(-torch.log(v)/theta-torch.log(v[b]))
            z_tilde = z_tilde.scatter(dim=-1, index=b, src=-torch.log(-torch.log(v[b])))
            # v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
            # z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
            return z_tilde
        ops_weights_z_tilde = _get_ops_weights_z_tilde(ops_weights_logits, ops_weights_b, ops_weights_v)
        self.ops_weights_logits = ops_weights_logits
        self.ops_weights_b = ops_weights_b
        self.ops_weights_softmax_z = torch.nn.functional.softmax(ops_weights_z/self.temperature, dim=-1)
        self.ops_weights_softmax_z_tilde = torch.nn.functional.softmax(ops_weights_z_tilde/self.temperature, dim=-1)
