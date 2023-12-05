import numpy as np
import torch
import torch.nn as nn

SQRT_CONSTL = 1e-6
SQRT_CONSTR = 1e8

def safe_sqrt(x, lbound=SQRT_CONSTL, rbound=SQRT_CONSTR):
    ''' Numerically safe version of TensorFlow sqrt '''
    return torch.sqrt(torch.clamp(x, lbound, rbound))

class Net(nn.Module):
    def __init__(self, x_dim=10, dropout=0, dim_in=16, dim_out=16):
        super(Net, self).__init__()

        activation=nn.ELU()

        self.rep_net = nn.Sequential(nn.Linear(x_dim, dim_in),
                                    activation,
                                    nn.Dropout(dropout),
                                    nn.Linear(dim_in, dim_in),
                                    activation,
                                    nn.Dropout(dropout),
                                    nn.Linear(dim_in, dim_in),
                                    activation,
                                    nn.Dropout(dropout))

        self.s_net = nn.Sequential(nn.Linear(dim_in+1, dim_out),
                                    activation,
                                    nn.Dropout(dropout),
                                    nn.Linear(dim_out, dim_out),
                                    activation,
                                    nn.Dropout(dropout),
                                    nn.Linear(dim_out, dim_out),
                                    activation,
                                    nn.Dropout(dropout),
                                    nn.Linear(dim_out, 1))

        self.y_net = nn.Sequential(nn.Linear(dim_in+2, dim_out),
                                    activation,
                                    nn.Dropout(dropout),
                                    nn.Linear(dim_out, dim_out),
                                    activation,
                                    nn.Dropout(dropout),
                                    nn.Linear(dim_out, dim_out),
                                    activation,
                                    nn.Dropout(dropout),
                                    nn.Linear(dim_out, 1))
        
    def reinit(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                nn.init.zeros_(module.bias.data)

    def output(self, x, t):
        h_rep = self.rep_net(x)
        h_rep_norm = h_rep / safe_sqrt(torch.sum(torch.square(h_rep), dim=1, keepdim=True))

        inputs = torch.cat([h_rep_norm, t], dim=1)
        s = self.s_net(inputs)
        # y = self.y_net(h_rep_norm)
        y_inputs = torch.cat([inputs, s], dim=1)
        y = self.y_net(y_inputs)

        return s, y

    def forward(self, x, t):
        h_rep = self.rep_net(x)
        h_rep_norm = h_rep / safe_sqrt(torch.sum(torch.square(h_rep), dim=1, keepdim=True))

        inputs = torch.cat([h_rep_norm, t], dim=1)
        s = self.s_net(inputs)
        # y = self.y_net(h_rep_norm)
        y_inputs = torch.cat([inputs, s], dim=1)
        y = self.y_net(y_inputs)

        return h_rep_norm, s, y
    