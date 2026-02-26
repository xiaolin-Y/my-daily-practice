import argparse
import torch
import torch.nn as nn
import numpy as np
import operator
from functools import reduce


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_pos_lst(size_lst, length):
    pos_lst = []
    for size in size_lst:
        nx, ny, nz = size
        pos_x = torch.linspace(0, length[0], nx).float().cuda().unsqueeze(-1)
        pos_y = torch.linspace(0, length[1], ny).float().cuda().unsqueeze(-1)
        pos_z = torch.linspace(0, length[2], nz).float().cuda().unsqueeze(-1)
        pos_lst.append([pos_x, pos_y, pos_z])
    return pos_lst

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.shape[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class ContinuityLoss(torch.nn.Module):
    def __init__(self, dx=1.0, dy=1.0, dz=1.0):
        super().__init__()
        self.dx, self.dy, self.dz = dx, dy, dz

    def forward(self, vel_field):
        # vel_field: (b, nx, ny, nz, 3) -> 假设前三通道是 u,v,w
        u = vel_field[..., 0]
        v = vel_field[..., 1]
        w = vel_field[..., 2]

        du_dx = torch.gradient(u, spacing=self.dx, dim=1)[0]
        dv_dy = torch.gradient(v, spacing=self.dy, dim=2)[0]
        dw_dz = torch.gradient(w, spacing=self.dz, dim=3)[0]

        divergence = du_dx + dv_dy + dw_dz
        return torch.mean(divergence**2)
    
    
# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c