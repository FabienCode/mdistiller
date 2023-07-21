import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_plans, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_plans, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion*planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample