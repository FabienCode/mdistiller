import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


class AreaDetection(nn.Module):
    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 num_cls=2):
        super().__init__()
        self.num_cls = num_cls
        self.in_channels = in_channels
        self.feat_channels = feat_channels

        self.heatmap_head = self._build_head(in_channels, feat_channels, num_cls)
        self.offset_head = self._build_head(in_channels, feat_channels, 2)
        self.wh_head = self._build_head(in_channels, feat_channels, 2)

    @staticmethod
    def _build_head(in_channels, feat_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, out_channels, kernel_size=1)
        )
        return layer

    def forward(self, x):
        center_heatmap_pred = self.heatmap_head(x).sigmoid()
        wh_pred = self.wh_head(x)
        offset_pred = self.offset_head(x)
        return center_heatmap_pred, wh_pred, offset_pred


def get_topk_from_featmap(scores, k=20):
    """
    Get top k positions from heatmap.
    Args:
        scores (torch.Tensor): heatmap scores, [B, C, H, W]
        k (int): number of topk
    Returns:
        tuple: topk scores, topk indexes, topk clses, topk_ys, topk_xs
        - topk scores (torch.Tensor): [B, k] Max scores of each topk keypoint.
        - topk indexes (torch.Tensor): [B, k] Indexes of each topk keypoint.
        - topk_clses (torch.Tensor): [B, k] Categories of each topk keypoint.
        - topk_ys (torch.Tensor): [B, k] y-coordinates of each topk keypoint.
        - topk_xs (torch.Tensor): [B, k] x-coordinates of each topk keypoint.
    """
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def _single_extract_area(heatmap, offset_pred, wh_pred, score_threshold=0.05):
    heatmap = F.softmax(heatmap, dim=1)
    max_pool = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    peaks = torch.where(heatmap == max_pool, 1, 0)
    peaks = torch.where(heatmap * peaks >= score_threshold, 1, 0)
    peak_locs = torch.nonzero(peaks)

    areas = []
    for loc in peak_locs:
        y, x = loc[0], loc[1]
        w, h = wh_pred[:, y, x]
        offset_y, offset_x = offset_pred[:, y, x]
        x1 = (x + offset_x - w / 2).clamp(min=0)
        y1 = (y + offset_y - h / 2).clamp(min=0)
        x2 = (x + offset_x + w / 2).clamp(max=(heatmap.shape[2] - 1))
        y2 = (y + offset_y + h / 2).clamp(max=(heatmap.shape[1] - 1))
        area = torch.tensor([x1, y1, x2, y2])

    return torch.stack(areas)


def extract_area(heatmaps, offset_preds, wh_preds, score_threshold=0.05, num_areas=256):
    areas = []
    for i in range(heatmaps.shape[0]):
        areas.append(_single_extract_area(heatmaps[i], offset_preds[i], wh_preds[i], score_threshold))

    return areas


def extract_regions(features, heatmap, wh_pred, offset_pred, score_threshold=0.05, resize_to_original=True):
    areas = extract_area(heatmap, offset_pred, wh_pred, score_threshold)
    # Initialize a list to hold the masks
    masks = []
    # Iterate over the images and areas
    for i in range(len(areas)):
        tmp_mask = []
        for j in range(areas[i].shape[0]):
            # Initialize a mask for this feature map
            mask = torch.zeros_like(features[i][j])
            x1, y1, x2, y2 = areas[i][j]
            mask[int(y1):int(y2), int(x1):int(x2)] = 1
            tmp_mask.append(mask)
        masks.append(torch.stack(tmp_mask))
    # min_num_areas = min([m.shape[0] for m in masks])
    # masks = [trim_tensor(m, min_num_areas) for m in masks]
    # masks = torch.stack(masks)

    # if resize_to_original:
    #     # Resize the masks to the original size
    #     masks = F.interpolate(masks, size=(features.shape[2], features.shape[3]))

    return masks


def trim_tensor(t, min_dim):
    return t[:min_dim]
