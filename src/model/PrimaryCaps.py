import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent # main directory, the parent of src
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from src.model.functions import squash


class PrimaryCaps(nn.Module):

    def __init__(self, in_channels, out_channels, out_caps_dim, kernel_size, stride, cuda_enabled=True, device='cuda:0', shared_weights=256, squash_approx=False):
        super(PrimaryCaps, self).__init__()

        self.C = in_channels
        self.M = out_channels
        self.Cout = out_caps_dim
        self.K = kernel_size
        self.U = stride

        self.shared_weights = shared_weights if shared_weights else 1
        self.squash_approx = squash_approx

        self.conv_units = nn.Conv2d(in_channels=in_channels,
                                    out_channels=int(
                                        out_channels*out_caps_dim / self.shared_weights),  # must be triple checked
                                    kernel_size=kernel_size,
                                    stride=stride)

        if cuda_enabled:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

    def forward(self, x):

        bs = x.size(0)
        uij = self.conv_units(x)  # bs  32*8  6  6
        uij = uij.repeat(1, self.shared_weights, 1, 1)

        uij = uij.view(bs, self.M, self.Cout, -1).permute(0, 1,
                                                          3, 2).contiguous().view(bs, -1, self.Cout)
        # bs  1152  8

        cij = torch.zeros(bs, uij.size(
            1), 1, device=self.device).fill_(1/uij.size(1))

        sj = cij * uij
        vj = squash(sj, dim=2, squash_approx=self.squash_approx)

        return vj
