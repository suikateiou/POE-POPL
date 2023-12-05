import numpy as np
import torch
import torch.nn as nn

SQRT_CONSTL = 1e-6
SQRT_CONSTR = 1e8


def safe_sqrt(x, lbound=SQRT_CONSTL, rbound=SQRT_CONSTR):
    ''' Numerically safe version of TensorFlow sqrt '''
    return torch.sqrt(torch.clamp(x, lbound, rbound))


class PolicyNet(nn.Module):
    def __init__(self, x_dim=10, dropout=0, dim_in=16, dim_out=16):
        super(PolicyNet, self).__init__()

        activation = nn.ELU()

        self.reg_net = nn.Sequential(nn.Linear(x_dim, dim_in),
                                     activation,
                                     nn.Dropout(dropout),
                                     nn.Linear(dim_in, dim_in),
                                     activation,
                                     nn.Dropout(dropout),
                                     nn.Linear(dim_in, dim_in),
                                     activation,
                                     nn.Dropout(dropout),
                                     nn.Linear(dim_out, 1))

    def reinit(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                nn.init.zeros_(module.bias.data)

    def output(self, x):
        t = self.reg_net(x)
        return t

    def forward(self, x):
        t = self.reg_net(x)
        return t
