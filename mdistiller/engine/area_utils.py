import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time

class AreaDetection(nn.Module):
    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 num_cls=2):
        super().__init__()
        self.num_cls = num_cls
        self.in_channels = in_channels
        self.feat_channels = feat_channels

        # AD
        self.heatmap_head = self._build_head(in_channels, feat_channels, num_cls)
        self.offset_head = self._build_head(in_channels, feat_channels, 2)
        self.wh_head = self._build_head(in_channels, feat_channels, 2)


    @staticmethod
    def _build_head(in_channels, feat_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, out_channels, kernel_size=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return layer

    def forward(self, x):
        center_heatmap_pred = self.heatmap_head(x).sigmoid()
        wh_pred = self.wh_head(x)
        offset_pred = self.offset_head(x)
        return center_heatmap_pred, wh_pred, offset_pred

def get_import_region(feature_map, center_heatmap_pred, wh_pred, offset_pred, k, kernel):
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
    batch_regions = torch.cat((batch_regions, batch_scores[..., None]), dim=-1)
    return batch_regions, batch_topk_clses

def extract_regions(features, heatmap, wh_pred, offset_pred, k, kernel):
    areas, topk_cls =  get_import_region(features, heatmap, wh_pred, offset_pred, k, kernel)
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

def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight, a=1)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight, a=1)
        m.bias.data.fill_(0.01)
