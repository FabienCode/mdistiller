import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
import torch.nn.functional as F
from mdistiller.engine.diffusion_utiles import cosine_beta_schedule, default, extract
from mdistiller.engine.mvkd_utils import Model

from transformers import CLIPProcessor, CLIPModel, ViltProcessor, ViltForQuestionAnswering
import numpy as np
from mdistiller.engine.light_diffusion import DiffusionModel, AutoEncoder
from mdistiller.engine.classes_datasets import CIFAR100_Labels, Imagenet_Labels


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class DLKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(DLKD, self).__init__(student, teacher)
        self.temperature = cfg.DLKD.TEMPERATURE
        self.ce_loss_weight = cfg.DLKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.DLKD.LOSS.KD_WEIGHT



    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
