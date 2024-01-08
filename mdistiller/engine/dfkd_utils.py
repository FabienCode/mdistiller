import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import numpy as np


class DifferentiableAugment(nn.Module):
    def __init__(self, sub_policy):
        super(DifferentiableAugment, self).__init__()
        self.sub_policy = sub_policy

    def forward(self, origin_images, trans_images, probability, probability_index, magnitude):
        index = sum(p_i.item() << i for i, p_i in enumerate(probability_index))
        com_image = 0
        images = origin_images
        adds = 0

        for selection in range(2 ** len(self.sub_policy)):
            trans_probability = 1
            for i in range(len(self.sub_policy)):
                if selection & (1 << i):
                    trans_probability = trans_probability * probability[i]
                    if selection == index:
                        images = images - magnitude[i]
                        adds = adds + magnitude[i]
                else:
                    trans_probability = trans_probability * (1 - probability[i])
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


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.DFKD.momentum
        self.network_weight_decay = args.DFKD.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.augment_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data.detach()
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data.detach() + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_model.set_augmenting(False)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        # unrolled_loss.backward()
        # dalpha = [v.grad for v in unrolled_model.augment_parameters()]
        # vector = [v.grad.data for v in unrolled_model.parameters()]
        # implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
        #
        # for g, ig in zip(dalpha, implicit_grads):
        #     g.data.sub_(eta, ig.data)

        unrolled_loss.backward()
        dalpha = []
        vector = [v.grad.data.detach() for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
        for ig in implicit_grads:
            dalpha += [-ig]

        for v, g in zip(self.model.augment_parameters(), dalpha):
            if v.grad is None:
                if not (g is None):
                    v.grad = Variable(g.data)
            else:
                if not (g is None):
                    v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).data.detach().norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        # grads_p = torch.autograd.grad(loss, self.model.augment_parameters(), retain_graph=True, allow_unused=True)
        grads_p = self.model.relax(loss)
        m_grads_p = torch.autograd.grad(loss, [self.model.augment_parameters()[2]], retain_graph=True, allow_unused=True)[0]
        if m_grads_p is None:
            m_grads_p = torch.zeros_like(self.model.augment_parameters()[2])
        grads_p.insert(2, m_grads_p)

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input, target)
        grads_n = self.model.relax(loss)
        m_grads_n = torch.autograd.grad(loss, [self.model.augment_parameters()[2]], retain_graph=True, allow_unused=True)[0]
        if m_grads_n is None:
            m_grads_n = torch.zeros_like(self.model.augment_parameters()[2])
        grads_n.insert(2, m_grads_n)
        # for x in grads_n:
        #     print(x.shape)

        for p, v in zip(self.model.parameters(), vector):
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
