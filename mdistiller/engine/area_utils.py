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
                 in_channels = 256,
                 feat_channels = 256,
                 num_cls = 2):
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
    
def extract_area(heatmap, offset_pred, wh_pred, score_threshold=0.5):
    # Apply softmax to convert the heatmap to probabilities
    heatmap = F.softmax(heatmap, dim=0)

    # Identify peaks in the heatmap
    # This can be done by applying a 3*3 max pooling and selecting pixels where the heatmap is equal to the max pooled heatmap
    max_pool = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    peaks = torch.where(heatmap == max_pool, 1, 0)

    # Discard peaks with a probability lower than the threshold
    peaks = torch.where(heatmap * peaks >= score_threshold, 1, 0)

    # Find the locations of the remaining peaks
    peak_locs = torch.nonzero(peaks)

    # Initialize a list to hold the importance area
    areas = []

    # Iterate over the peak locations
    for loc in peak_locs:
        y, x = loc[0], loc[1]

        # Get the width and height of the box and the offset from the predictions
        w, h = wh_pred[:, y, x]
        offset_y, offset_x = offset_pred[:, y, x]

        # Calculate the importance area and add them to the list
        area_x1 = (x + offset_x - w / 2).clamp(min=0)
        area_y1 = (y + offset_y - h / 2).clamp(min=0)
        area_x2 = (x + offset_x + w / 2).clamp(max=heatmap.shape[1]-1)
        area_y2 = (y + offset_y + h / 2).clamp(max=heatmap.shape[0]-1)

        areas.append(torch.tensor([area_x1, area_y1, area_x2, area_y2]))
    
    # stack all the areas together
    areas = torch.stack(areas)

    return areas

def extract_regions(features, heatmap, wh_pred, offset_pred, score_threshold=0.5, resize_to_original=False):
    # Assume input tensor are of shape [batch, channel, height, width]

    # Get the areas from the heatmap and predictions
    areas = extract_area(heatmap, offset_pred, wh_pred, score_threshold)

    # Initialize a list to hold the masks
    masks = []

    # Iterate over the images and areas
    for i in range(features.shape[0]):
        # Get the current input feature
        feature = features[i]

        # Initialize a mask for this feature map
        mask = torch.zeros_like(feature)

        # Get the area for this input
        input_areas = areas[i]

        for area in input_areas:
            x1, y1, x2, y2 = area
            mask[y1:y2, x1:x2] = 1
        masks.append(mask)

    # Stack the masks together
    masks = torch.stack(masks)

    if resize_to_original:
        # Resize the masks to the original size
        masks = F.interpolate(masks, size=(features.shape[2], features.shape[3]))
    
    return masks
        