import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
# from mdistiller.engine.area_utils import get_import_region as saliency_bbox


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
        weak_saliency = get_saliency_region(heat_map_t_w, wh_t_w, offset_t_w, self.topk_area, self.kernel_size)
        strong_saliency = get_saliency_region(head_map_t_s, wh_t_s, offset_t_s, self.topk_area, self.kernel_size)
        f_t_w_aug = replace_bbox_values(f_t_w, f_t_s, strong_saliency)
        f_t_s_aug = replace_bbox_values(f_t_s, f_t_w, weak_saliency)

        # for i in range(saliency_t_w_b.shape[1]):
        #     saliency_tmp_w = saliency_t_w_b[:, i, :]
        #     saliency_tmp_s = saliency_t_s_b[:, i, :]
        #     f_t_w_aug, f_t_s_aug = aug_feat(f_t_w_aug, f_t_s_aug, saliency_tmp_w, saliency_tmp_s)
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

def get_saliency_region(center_heatmap_pred, wh_pred, offset_pred, k, kernel):
    height, width = center_heatmap_pred.shape[2:]

    center_heatmap_pred = get_local_maximum(center_heatmap_pred, kernel=kernel)
    *batch_region, topk_ys, topk_xs = get_topk_from_heatmap(center_heatmap_pred, k=k)
    batch_scores, batch_index, batch_topk_clses, = batch_region

    wh = transpose_and_gather_feat(wh_pred, batch_index)
    offset = transpose_and_gather_feat(offset_pred, batch_index)
    topk_xs = topk_xs + offset[..., 0]
    topk_ys = topk_ys + offset[..., 1]
    tl_x = (topk_xs - wh[..., 0] / 2).clamp(min=0)
    tl_y = (topk_ys - wh[..., 1] / 2).clamp(min=0)
    br_x = (topk_xs + wh[..., 0] / 2).clamp(max=(width - 1))
    br_y = (topk_ys + wh[..., 1] / 2).clamp(max=(height - 1))

    batch_regions = torch.stack([tl_x.int(), tl_y.int(), br_x.int(), br_y.int()], dim=2)
    # return tl_x.int(), tl_y.int(),  br_x.int(), br_y.int()
    return batch_regions


def extract_regions(features, heatmap, wh_pred, offset_pred, k, kernel):
    areas, topk_cls = get_saliency_region(features, heatmap, wh_pred, offset_pred, k, kernel)
    # Initialize a list to hold the masks
    masks = []
    # Iterate over the images and areas
    for i in range(areas.shape[0]):
        tmp_mask = []
        for j in range(k):
            # Initialize a mask for this feature map
            mask = torch.zeros_like(features[i][j])
            x1, y1, x2, y2, _ = areas[i][j]
            mask[int(y1):int(y2)+1, int(x1):int(x2)+1] = 1
            tmp_mask.append(mask)
        masks.append(torch.stack(tmp_mask))
    return masks, areas[..., -1]


def get_local_maximum(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(scores, k=20):
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    # topk_clses = topk_inds // (height * width)
    topk_clses = torch.div(topk_inds, height * width, rounding_mode='floor')
    topk_inds = topk_inds % (height * width)
    # topk_ys = topk_inds // width
    topk_ys = torch.div(topk_inds, width, rounding_mode='floor')
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat


def replace_bbox_values(image_source, image_target, bounding_boxes):
    # 提取 bounding box 的坐标
    x1 = bounding_boxes[:, :, 0:1]
    y1 = bounding_boxes[:, :, 1:2]
    x2 = bounding_boxes[:, :, 2:3]
    y2 = bounding_boxes[:, :, 3:4]

    # 创建索引张量
    batch_idx = torch.arange(image_source.size(0)).unsqueeze(1)  # 生成 batch 索引
    grid_x, grid_y = torch.meshgrid(torch.arange(image_source.size(2)), torch.arange(image_source.size(3)))  # 生成网格坐标

    # 使用 torch.where 创建掩码
    mask_x = (grid_x >= x1) & (grid_x <= x2)
    mask_y = (grid_y >= y1) & (grid_y <= y2)
    mask = mask_x & mask_y

    # 使用高级索引将对应位置的值替换为目标图像相同位置的值
    mask_expanded = mask.unsqueeze(1).expand_as(image_source)  # 扩展维度以匹配源图像的维度
    image_source[mask_expanded] = image_target[mask_expanded]

    return image_source

