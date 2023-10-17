import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        # self.bn = nn.BatchNorm2d(t_C)
        self.ln = nn.LayerNorm(t_C)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(t_C, t_C, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(t_C)
        self.ln2 = nn.LayerNorm(t_C)
        self.conv_3 = nn.Conv2d(t_C, t_C, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(t_C)
        self.ln3 = nn.LayerNorm(t_C)

    # def forward(self, x):
    #     x = self.conv(x)
    #     # if self.use_relu:
    #     #     return self.relu(self.bn(x))
    #     # else:
    #     #     return self.bn(x)
    #     if self.use_relu:
    #         x = self.relu(self.bn(x))
    #     else:
    #         x = self.bn(x)
    #     x = self.conv_2(x)
    #     if self.use_relu:
    #         x = self.relu(self.bn2(x))
    #     else:
    #         x = self.bn2(x)
    #     x = self.conv_3(x)
    #     if self.use_relu:
    #         return self.relu(self.bn3(x))
    #     else:
    #         return self.bn3(x)

    def forward(self, x):
        x = self.conv(x)
        # if self.use_relu:
        #     return self.relu(self.bn(x))
        # else:
        #     return self.bn(x)
        if self.use_relu:
            x = self.relu(self.ln(x))
        else:
            x = self.ln(x)
        x = self.conv_2(x)
        if self.use_relu:
            x = self.relu(self.ln2(x))
        else:
            x = self.ln2(x)
        x = self.conv_3(x)
        if self.use_relu:
            return self.relu(self.ln3(x))
        else:
            return self.ln3(x)
        
class ConvRegE(nn.Module):
    """Convolutional regression"""

    def __init__(self, in_channel, out_channel, use_relu=False):
        super(ConvRegE, self).__init__()
        self.use_relu = use_relu
        # s_N, s_C, s_H, s_W = s_shape
        # t_N, t_C, t_H, t_W = t_shape
        # if s_H == 2 * t_H:
        #     self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        # elif s_H * 2 == t_H:
        #     self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        # elif s_H >= t_H:
        #     self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        # else:
        #     raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv_3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.layernorm = nn.GroupNorm(num_groups=1, num_channels=out_channel, affine=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv_1(x)))
        x = self.relu(self.bn2(self.conv_2(x)))
        x = self.relu(self.bn3(self.conv_3(x)))
        return x


def get_feat_shapes(student, teacher, input_size):
    data = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        _, feat_s = student(data)
        _, feat_t = teacher(data)
    feat_s_shapes = [f.shape for f in feat_s["feats"]]
    feat_t_shapes = [f.shape for f in feat_t["feats"]]
    return feat_s_shapes, feat_t_shapes
