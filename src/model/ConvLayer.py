import math
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent # main directory, the parent of src
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, cuda_enabled=True, device='cuda:0', shared_weights=256, shared_bias=256):
        super(ConvLayer, self).__init__()

        if cuda_enabled:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        self.shared_weights = shared_weights if shared_weights else 1
        self.shared_bias = shared_bias if shared_bias else 1  # included with weight sharing
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=int(
                                  out_channels / self.shared_weights),
                              kernel_size=kernel_size,
                              stride=stride)

        self.relu = nn.ReLU()

    def forward(self, x):
        out_conv = self.conv(x)
        out_conv = out_conv.repeat(1, self.shared_weights, 1, 1)
        out_relu = self.relu(out_conv)

        return out_relu
