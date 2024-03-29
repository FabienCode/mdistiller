import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import numpy as np


class DifferentiableAugment(nn.Module):
    def __init__(self, sub_policy):
        super(DifferentiableAugment, self).__init__()
        self.sub_policy = sub_policy

    def forward(self, origin_images, probability_b, magnitude):
        device = origin_images.device
        images = origin_images.to(device)
        probability_b = probability_b.to(device)
        magnitude = magnitude.to(device)
        adds = 0
        for i in range(len(self.sub_policy)):
            if probability_b[i].item() != 0.0:
                images = images - magnitude[i]
                adds = adds + magnitude[i]
        images = images.detach() + adds
        return images


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

    def forward(self, origin_images, probabilities_b, magnitudes, weights_b):
        return self._ops[weights_b.item()](origin_images, probabilities_b[weights_b.item()], magnitudes[weights_b.item()])


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.DFKD.momentum
        self.network_weight_decay = args.DFKD.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.module.augment_parameters(),
                                          lr=args.DFKD.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.DFKD.arch_weight_decay)

    def _compute_unrolled_model(self, image, target, eta, network_optimizer, epoch):
        pred, loss_dict = self.model.module.forward_train(image, target, epoch=epoch)
        loss = sum(loss_dict.values())
        theta = _concat(self.model.module.get_learnable_parameters()).data.detach()
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        grad = torch.autograd.grad(loss, self.model.module.get_learnable_parameters())
        # grad = torch.autograd.grad(loss, self.model.module.augment_parameters())
        dtheta = _concat(grad).data.detach() + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, image, target, eta, network_optimizer, epoch, unrolled=True):
        self.optimizer.zero_grad()
        self._backward_step_unrolled(image, target, eta, network_optimizer, epoch)
        self.optimizer.step()

    def _backward_step_unrolled(self, input_train, target_train, eta, network_optimizer, epoch):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer, epoch)
        unrolled_model.module.set_augmenting(False)
        pred, loss_dict = unrolled_model.module.forward_train(input_train, target_train, epoch=epoch)
        unrolled_loss = sum(loss_dict.values())

        unrolled_loss.backward()
        dalpha = []
        vector = [v.grad.data.detach() for k, v in unrolled_model.named_parameters() if 'teacher' not in k]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train, epoch=epoch)
        for ig in implicit_grads:
            dalpha += [-ig]

        for v, g in zip(self.model.module.augment_parameters(), dalpha):
            if v.grad is None:
                if not (g is None):
                    v.grad = Variable(g.data)
            else:
                if not (g is None):
                    v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.module.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            if 'teacher' not in k:
                v_length = np.prod(v.size())
                params[k] = theta[offset: offset + v_length].view(v.size())
                offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, epoch, r=1e-2):
        R = r / _concat(vector).data.detach().norm()
        for p, v in zip(self.model.module.get_learnable_parameters(), vector):
            p.data.add_(R, v)
        # loss = self.model._loss(input, target)
        pred, loss_dict = self.model.module.forward_train(input, target, epoch=epoch)
        loss = sum(loss_dict.values())
        # grads_p = torch.autograd.grad(loss, self.model.augment_parameters(), retain_graph=True, allow_unused=True)
        grads_p = self.model.module.relax(loss)
        m_grads_p = torch.autograd.grad(loss, [self.model.module.augment_parameters()[2]], retain_graph=True, allow_unused=True)[0]
        if m_grads_p is None:
            m_grads_p = torch.zeros_like(self.model.module.augment_parameters()[2])
        grads_p.insert(2, m_grads_p)

        for p, v in zip(self.model.module.get_learnable_parameters(), vector):
            p.data.sub_(2 * R, v)
        pred, loss_dict = self.model.module.forward_train(input, target, epoch=epoch)
        loss = sum(loss_dict.values())
        grads_n = self.model.module.relax(loss)
        m_grads_n = torch.autograd.grad(loss, [self.model.module.augment_parameters()[2]], retain_graph=True, allow_unused=True)[0]
        if m_grads_n is None:
            m_grads_n = torch.zeros_like(self.model.module.augment_parameters()[2])
        grads_n.insert(2, m_grads_n)
        # for x in grads_n:
        #     print(x.shape)

        for p, v in zip(self.model.module.get_learnable_parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


class QFunc(torch.nn.Module):
    '''Control variate for RELAX'''

    def __init__(self, num_latents, hidden_size=100):
        super(QFunc, self).__init__()
        self.h1 = torch.nn.Linear(num_latents, hidden_size)
        self.nonlin = torch.nn.Tanh()
        self.out = torch.nn.Linear(hidden_size, 1)

    def forward(self, p, w):
        # the multiplication by 2 and subtraction is from toy.py...
        # it doesn't change the bias of the estimator, I guess
        # print(p, w)
        z = torch.cat([p, w.unsqueeze(dim=-1)], dim=-1)
        z = z.reshape(-1)
        # print(z)
        z = self.h1(z * 2. - 1.)
        # print(z)
        z = self.nonlin(z)
        # print(z)
        z = self.out(z)
        # print(z)
        return z
