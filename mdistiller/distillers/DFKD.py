import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes

from mdistiller.engine.dfkd_primitives import sub_policies
import random
from mdistiller.engine.dfkd_utils import MixedAugment
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
