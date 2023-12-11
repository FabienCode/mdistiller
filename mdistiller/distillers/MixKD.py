import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
from mdistiller.engine.area_utils import get_import_region as saliency_bbox


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd


class MixKD(Distiller):

    def __init__(self, student, teacher, cfg):
        super(MixKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.MixKD.LOSS.CE_WEIGHT
        self.feat_loss_weight_ori = cfg.MixKD.LOSS.ORI_FEAT_WEIGHT
        self.feat_loss_weight_aug = cfg.MixKD.LOSS.AUG_FEAT_WEIGHT
        self.hint_layer = cfg.MixKD.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )
        self.saliency_det = SaliencyAreaDetection(int(feat_t_shapes[self.hint_layer][1]),
                                                  int(feat_t_shapes[self.hint_layer][1]),
                                                  2, 100)

        self.kernel_size = cfg.MixKD.KERNEL_SIZE
        self.topk_area = cfg.MixKD.TOPK_AREA
    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.parameters()) + \
            list(self.saliency_det.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        for p in self.saliency_det.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, feature_student_weak = self.student(image_weak)
        logits_student_strong, feature_student_strong = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, feature_teacher_weak = self.teacher(image_weak)
            logits_teacher_strong, feature_teacher_strong = self.teacher(image_strong)

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))

        f_s_w = feature_student_weak["feats"][self.hint_layer]
        f_s_s = feature_student_strong["feats"][self.hint_layer]
        f_t_w = feature_teacher_weak["feats"][self.hint_layer]
        f_t_s = feature_teacher_strong["feats"][self.hint_layer]
        loss_feat_ori = F.mse_loss(f_s_w, f_t_w) + F.mse_loss(f_s_w, f_t_s)
        # saliency compute
        heat_map_t_w, wh_t_w, offset_t_w = self.saliency_det(f_t_w)
        head_map_t_s, wh_t_s, offset_t_s = self.saliency_det(f_t_s)
        saliency_t_w_b, _ = saliency_bbox(heat_map_t_w, wh_t_w, offset_t_w, self.topk_area, self.kernel_size)
        saliency_t_s_b, _ = saliency_bbox(head_map_t_s, wh_t_s, offset_t_s, self.topk_area, self.kernel_size)
        f_t_w_aug = f_t_w.clone()
        f_t_s_aug = f_t_s.clone()
        for i in range(saliency_t_w_b.shape[1]):
            saliency_tmp_w = saliency_t_w_b[:, i, :]
            saliency_tmp_s = saliency_t_s_b[:, i, :]
            f_t_w_aug, f_t_s_aug = aug_feat(f_t_w_aug, f_t_s_aug, saliency_tmp_w, saliency_tmp_s)
        # loss_kd = kd_loss(logits_student_weak, logits_teacher_weak, 4) + kd_loss(
        #     logits_student_weak, logits_teacher_strong, 4
        # )
        loss_feat_aug = F.mse_loss(f_s_w, f_t_w_aug) + F.mse_loss(f_s_w, f_t_s_aug)
        loss_feat = self.feat_loss_weight_aug * loss_feat_aug + self.feat_loss_weight_ori * loss_feat_ori
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_feat_weak": loss_feat
        }
        return logits_student_weak, losses_dict


class SaliencyAreaDetection(nn.Module):
    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 num_cls=2,
                 cls=100,
                 thresh=0.8):
        super().__init__()
        self.num_cls = num_cls
        self.in_channels = in_channels
        self.feat_channels = feat_channels

        # AD
        self.heatmap_head = self._build_head(in_channels, feat_channels, num_cls)
        self.offset_head = self._build_head(in_channels, feat_channels, 2)
        self.wh_head = self._build_head(in_channels, feat_channels, 2)



        self.softmax = nn.Softmax(dim=1)
        # self.thresh_pred = nn.Linear(cls, 1)
        # self.gate = nn.Parameter(torch.tensor(-2.0), requires_grad=False)
        self.thresh = thresh
        self.sig = nn.Sigmoid()

    # test
    @staticmethod
    def _build_head(in_channels, feat_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        return layer

    def forward(self, x):
        heatmap_pred = self.heatmap_head(x)
        center_heatmap_pred = heatmap_pred.sigmoid()
        wh_pred = self.wh_head(x)
        b, _, h, w = wh_pred.shape
        offset_pred = self.offset_head(x)
        return center_heatmap_pred, wh_pred, offset_pred


def aug_feat(feature_weak, feature_strong, region_w, region_s):
    b, c, h, w = feature_weak.shape
    for batch_idx in range(b):
        w_x1, w_y1, w_x2, w_y2, _ = region_w[batch_idx].int()
        s_x1, s_y1, s_x2, s_y2, _ = region_s[batch_idx].int()
        for channel in range(c):
            feature_weak[batch_idx, channel, s_x1:s_x2, s_y1:s_y2] = feature_strong[batch_idx, channel, s_x1:s_x2, s_y1:s_y2].clone()
            feature_strong[batch_idx, channel, w_x1:w_x2, w_y1:w_y2] = feature_weak[batch_idx, channel, w_x1:w_x2, w_y1:w_y2].clone()
    return feature_weak, feature_strong


# def aug_feat(feature_weak, feature_strong, region_w, region_s):
#     # 使用高效的张量操作而非循环
#     b, c, h, w = feature_weak.shape
#     mask_w = torch.zeros(b, h, w, dtype=torch.bool)
#     mask_s = torch.zeros(b, h, w, dtype=torch.bool)
#
#     for batch_idx in range(b):
#         w_x1, w_y1, w_x2, w_y2, _ = region_w[batch_idx].int()
#         s_x1, s_y1, s_x2, s_y2, _ = region_s[batch_idx].int()
#         mask_w[batch_idx, s_y1:s_y2, s_x1:s_x2] = True
#         mask_s[batch_idx, w_y1:w_y2, w_x1:w_x2] = True
#
#     # 交换区域
#     feature_weak_clone = feature_weak.clone()
#     feature_weak[mask_w] = feature_strong[mask_w]
#     feature_strong[mask_s] = feature_weak_clone[mask_s]
#
#     return feature_weak, feature_strong
