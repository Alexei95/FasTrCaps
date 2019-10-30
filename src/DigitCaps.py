import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from functions import squash


class DigitCaps(nn.Module):

    def __init__(self, in_dim, out_dim, in_caps_dim, out_caps_dim, iter, cuda_enabled=True, device='cuda:0', shared_weights=36, squash_approx=False):
        super(DigitCaps, self).__init__()

        self.C = in_dim
        self.M = out_dim
        self.Cin = in_caps_dim
        self.Cout = out_caps_dim
        self.iter = iter

        if cuda_enabled:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        self.shared_weights = shared_weights if shared_weights else 1
        self.squash_approx = squash_approx

        self.weight = nn.Parameter(torch.empty(
            1, int(self.C / self.shared_weights), self.M, self.Cout, self.Cin))  # 1  1152  10  16  8
        self.bias = nn.Parameter(torch.empty(
            1, 1, self.M, self.Cout, 1))  # 1 1 10 16 1

        b = math.sqrt(1/in_caps_dim)
        a = -1. * b

        init.uniform_(self.weight, a=a, b=b)
        init.uniform_(self.bias, a=a, b=b)

        self.relu = nn.ReLU()

    def forward(self, x):

        # x:  bs  1152  8
        # y:  bs  10    16
        bs = x.size(0)

        x = x.unsqueeze(2).unsqueeze(4)   # bs  1152  1  8  1

        weights = self.weight.repeat(1, self.shared_weights, 1, 1, 1)

        u_hat = torch.matmul(weights, x)  # bs 1152  10  16  1

        bij = torch.zeros(bs, self.C, self.M, 1,
                          device=self.device)  # bs  1152  10  1
        cij = torch.zeros(bs, self.C, self.M, 1, 1,
                          device=self.device).fill_(1/self.M)

        for iter_ in range(self.iter):
            sj = (cij * u_hat).sum(dim=1, keepdim=True) + \
                self.bias  # bs   1   10  16 1

            # bs 1 10 16 1
            vj = squash(sj, dim=3, squash_approx=self.squash_approx)

            if iter_ < self.iter-1.:

                bij = bij + \
                    (torch.matmul(u_hat.transpose(3, 4), vj)).squeeze(4)
                cij = (F.softmax(bij, dim=2)).unsqueeze(
                    4)  # bs  1152  10  1  1

        return vj.squeeze(4).squeeze(1)
