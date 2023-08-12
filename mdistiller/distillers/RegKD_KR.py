import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes

from mdistiller.engine.area_utils import AreaDetection, extract_regions, RegKD_pred
from mdistiller.engine.kd_loss import mask_logits_loss, dkd_loss
from mdistiller.models.cifar.resnet import BasicBlock, Bottleneck


class RegKD_KR(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(RegKD_KR, self).__init__(student, teacher)
        self.cfg = cfg
        self.ce_loss_weight = cfg.RegKD.CE_WEIGHT
        self.alpha = cfg.RegKD.ALPHA
        self.beta = cfg.RegKD.BETA
        self.temperature = cfg.RegKD.T
        self.warmup = cfg.RegKD.WARMUP

        self.channel_weight = cfg.RegKD.CHANNEL_KD_WEIGHT
        self.area_weight = cfg.RegKD.AREA_KD_WEIGHT
        self.size_reg_weight = cfg.RegKD.SIZE_REG_WEIGHT

        self.shapes = cfg.RegKD.SHAPES
        self.out_shapes = cfg.RegKD.OUT_SHAPES
        in_channels = cfg.RegKD.IN_CHANNELS
        out_channels = cfg.RegKD.OUT_CHANNELS

        self.area_num = cfg.RegKD.AREA_NUM
        self.hint_layer = cfg.RegKD.HINT_LAYER
        self.shapes = cfg.RegKD.SHAPES
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.RegKD.INPUT_SIZE
        )
        self.stu_preact = cfg.RegKD.STU_PREACT
        # self.conv_reg = ConvReg(
        #     feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        # )
        # self.area_det = RegKD_pred(int(feat_t_shapes[self.hint_layer][1]),
        #                            int(feat_t_shapes[self.hint_layer][1]), 2, 100, cfg.RegKD.LOGITS_THRESH)

        # self.conv_reg = nn.ModuleList()
        # for i in range(4):
        #     self.conv_reg.append(ConvReg(feat_s_shapes[i], feat_t_shapes[i]))
        self.area_det = nn.ModuleList()
        for i in range(4):
            self.area_det.append(RegKD_pred(out_channels[i], out_channels[i], 2, 100, cfg.RegKD.LOGITS_THRESH))
        self.channel_mask = cfg.RegKD.CHANNEL_MASK

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

        # self.logits_thresh = cfg.RegKD.LOGITS_THRESH

    #
    # # list(self.score_norm.parameters())
    # def get_learnable_parameters(self):
    #     return super().get_learnable_parameters() + list(self.area_det.parameters()) + \
    #         list(self.conv_reg.parameters()) + list(self.abfs.parameters())
    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.area_det.parameters()) + \
            list(self.abfs.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.area_det.parameters():
            num_p += p.numel()
        # for p in self.conv_reg.parameters():
        #     num_p += p.numel()
        for p in self.abfs.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)

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
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_reviewkd = (
            self.area_weight
            * min(kwargs["epoch"] / self.warmup, 1.0)
            * hcl_loss(results, features_teacher)
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_reviewkd,
        }
        return logits_student, losses_dict


def aaloss(feature_student,
           feature_teacher,
           masks,
           scores):
    masks_stack = torch.stack(masks)
    # scores_expand = scores.unsqueeze(-1).unsqueeze(-1).expand_as(masks_stack)
    # weight_masks = masks_stack * scores_expand
    s_masks = masks_stack.sum(1)
    loss = F.mse_loss(feature_student * s_masks.unsqueeze(1), feature_teacher * s_masks.unsqueeze(1))
    return loss


def norm_tensor(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


def calculate_channel_importance(conv_layer):
    weights = conv_layer.weight.data
    l1_norm = weights.abs().sum(dim=(1, 2, 3))
    return l1_norm


def calculate_block_importance_and_mask(model, percentile=80):
    block_importance = {}
    block_mask = {}

    for name, module in model.named_modules():
        # 判断是否是BasicBlock或Bottleneck，这两种类型的模块都包含有多个卷积层
        if isinstance(module, (BasicBlock, Bottleneck)):
            # 获取block内最后一个卷积层，对于BasicBlock是conv2，对于Bottleneck是conv3
            last_conv = module.conv2 if isinstance(module, BasicBlock) else module.conv3
            # 计算该卷积层对应的通道重要性
            importance = calculate_channel_importance(last_conv)
            # 存储结果
            block_importance[name] = importance

            # 计算阈值
            threshold = np.percentile(importance.cpu().detach().numpy(), percentile)
            # 创建掩码，如果通道的重要性大于阈值，那么掩码的值为1，否则为0
            mask = (importance > threshold).float()
            # 存储结果
            block_mask[name] = mask

    return block_importance, block_mask


def compute_fc_importance(fc_layer):
    importance = torch.norm(fc_layer.weight.data, p=1, dim=1)
    return importance


def prune_fc_layer(fc_layer, percentage):
    fc_importance = compute_fc_importance(fc_layer)
    _, sorted_index = torch.sort(fc_importance)
    num_prune = int(len(sorted_index) * percentage)
    prune_index = sorted_index[:num_prune]
    mask = torch.ones(len(sorted_index), dtype=torch.bool)
    mask[prune_index] = 0
    return mask


def mask_kd_loss(logits_student, logits_teacher, temperature, mask=None):
    if mask is not None:
        logits_student = logits_student * mask
        logits_teacher = logits_teacher * mask
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def cat_mask_kd_loss(logits_student, logits_teacher, target, temperature, alpha, beta, mask=None):
    # if mask is not None:
    #     logits_student = logits_student * mask
    #     logits_teacher = logits_teacher * mask
    gt_mask = mask
    other_mask = ~mask
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = F.kl_div(log_pred_student, pred_teacher, size_average=False) * (temperature ** 2) / target.shape[0]
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


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