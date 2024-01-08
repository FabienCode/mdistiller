import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable


class DifferentiableAugment(nn.Module):
    def __init__(self, sub_policy):
        super(DifferentiableAugment, self).__init__()
        self.sub_policy = sub_policy

    def forward(self, origin_images, trans_images, probability, probability_index, magnitude):
        index = sum( p_i.item()<<i for i, p_i in enumerate(probability_index))
        com_image = 0
        images = origin_images
        adds = 0

        for selection in range(2**len(self.sub_policy)):
            trans_probability = 1
            for i in range(len(self.sub_policy)):
                if selection & (1<<i):
                    trans_probability = trans_probability * probability[i]
                    if selection == index:
                        images = images - magnitude[i]
                        adds = adds + magnitude[i]
                else:
                    trans_probability = trans_probability * ( 1 - probability[i] )
            if selection == index:
                images = images.detach() + adds
                com_image = com_image + trans_probability * images
            else:
                com_image = com_image + trans_probability

        # com_image = probability * trans_images + (1 - probability) * origin_images
        return com_image


class MixedAugment(nn.Module):
    def __init__(self, sub_policies):
        super(MixedAugment, self).__init__()
        self.sub_policies = sub_policies
        self._compile(sub_policies)

    def _compile(self, sub_polices):
        self._ops = nn.ModuleList()
        self._nums = len(sub_polices)
        for sub_policy in sub_polices:
            ops = DifferentiableAugment(sub_policy)
            self._ops.append(ops)

    def forward(self, origin_images, trans_images_list, probabilities, probabilities_index, magnitudes, weights, weights_index):
        trans_images = trans_images_list
        return sum(w * op(origin_images, trans_images, p, p_i, m) if weights_index.item() == i else w
                   for i, (p, p_i, m, w, op) in
                   enumerate(zip(probabilities, probabilities_index, magnitudes, weights, self._ops)))

class MixedAugment(nn.Module):
    def __init__(self, sub_policies):
        super(MixedAugment, self).__init__()
        self.sub_policies = sub_policies
        self._compile(sub_policies)

    def _compile(self, sub_polices):
        self._ops = nn.ModuleList()
        self._nums = len(sub_polices)
        for sub_policy in sub_polices:
            ops = DifferentiableAugment(sub_policy)
            self._ops.append(ops)

    def forward(self, origin_images, trans_images_list, probabilities, probabilities_index, magnitudes, weights, weights_index):
        trans_images = trans_images_list
        return sum(w * op(origin_images, trans_images, p, p_i, m) if weights_index.item() == i else w
                   for i, (p, p_i, m, w, op) in
                   enumerate(zip(probabilities, probabilities_index, magnitudes, weights, self._ops)))