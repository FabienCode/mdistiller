import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms.transforms import Compose

random_mirror = True

def apply_affine_transformation(feature, matrix):
    n, c, h, w = feature.size()
    # 确保 theta 是一个批次大小匹配的张量
    theta = matrix.unsqueeze(0).repeat(n, 1, 1).to(feature.device)
    grid = F.affine_grid(theta, feature.size(), align_corners=False)
    new_feature = F.grid_sample(feature, grid, align_corners=False)
    return new_feature


def ShearX(feature, v):  # [-0.3, 0.3]
    n, c, h, w = feature.size()
    # 使用 torch.tensor() 创建矩阵并确保它是浮点类型的
    matrix = torch.tensor([[1, v, 0], [0, 1, 0]], dtype=torch.float, device=feature.device)
    return apply_affine_transformation(feature, matrix)


def ShearY(feature, v):  # [-0.3, 0.3]
    n, c, h, w = feature.size()
    matrix = torch.tensor([[1, 0, 0], [v, 1, 0]], dtype=torch.float, device=feature.device)
    return apply_affine_transformation(feature, matrix)


def TranslateX(feature, v):  # [-0.45, 0.45]
    n, _, h, _ = feature.size()
    tx = v * h
    matrix = torch.tensor([[1, 0, tx], [0, 1, 0]], dtype=torch.float)
    return apply_affine_transformation(feature, matrix)

def TranslateY(feature, v):  # [-0.45, 0.45]
    n, _, _, w = feature.size()
    ty = v * w
    matrix = torch.tensor([[1, 0, 0], [0, 1, ty]], dtype=torch.float)
    return apply_affine_transformation(feature, matrix)

def TranslateXAbs(feature, v):  # [-150, 150]
    tx = v
    matrix = torch.tensor([[1, 0, tx], [0, 1, 0]], dtype=torch.float)
    return apply_affine_transformation(feature, matrix)

def TranslateYAbs(feature, v):  # [-150, 150]
    ty = v
    matrix = torch.tensor([[1, 0, 0], [0, 1, ty]], dtype=torch.float)
    return apply_affine_transformation(feature, matrix)

def Rotate(feature, v):  # [-30, 30]
    n, _, h, w = feature.size()
    angle = v * torch.pi / 180  # Convert degrees to radians
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    matrix = torch.tensor([[cos_a, sin_a, 0], [-sin_a, cos_a, 0]], dtype=torch.float)
    return apply_affine_transformation(feature, matrix)



def Cutout(feature, v):  # [0, 60] => percentage: [0, 0.2]
    n, c, h, w = feature.size()
    cutout_size = [int(v * h), int(v * w)]

    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - cutout_size[0] // 2, 0, h)
    y2 = np.clip(y + cutout_size[0] // 2, 0, h)
    x1 = np.clip(x - cutout_size[1] // 2, 0, w)
    x2 = np.clip(x + cutout_size[1] // 2, 0, w)

    mask = torch.ones((n, c, h, w), device=feature.device, dtype=feature.dtype)
    mask[:, :, y1:y2, x1:x2] = 0
    new_feature = feature * mask
    return new_feature




def Noise(feature, noise_level=0.05):  # [0, 0.4]
    noise = torch.randn_like(feature) * noise_level
    return feature + noise


def ChannelShuffle(feature, v):  # [0, 0.4]
    n, c, h, w = feature.size()
    idx = list(range(c))
    random.shuffle(idx)
    idx = torch.tensor(idx, device=feature.device)
    return feature[:, idx, :, :]


def ChannelDrop(feature, dropout_ratio=0.5):
    n, c, h, w = feature.size()
    mask = torch.rand(c, device=feature.device) > dropout_ratio
    mask = mask.view(1, c, 1, 1).expand_as(feature)
    return feature * mask


def augment_list(for_autoaug=True):  # 16 oeprations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (Cutout, 0, 0.2),  # 5
        (Noise, 0, 0.1),  # 6
        (ChannelShuffle, 0, 1),  # 7
        (ChannelDrop, 0, 0.5),  # 8
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    # if for_autoaug:
    #     l += [
    #         (CutoutAbs, 0, 20),  # compatible with auto-augment
    #         (Posterize2, 0, 4),  # 9
    #         (TranslateXAbs, 0, 10),  # 9
    #         (TranslateYAbs, 0, 10),  # 9
    #     ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


def get_augment(name):
    return augment_dict[name]


def apply_augment(feat, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(feat.clone(), level * (high - low) + low)


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


