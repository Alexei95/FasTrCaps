import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import mask


class Decoder(nn.Module):

    def __init__(self, in_dim, l1_dim, l2_dim, out_dim, cuda_enabled=True, device='cuda:0', small_decoder=False):

        super(Decoder, self).__init__()

        self.cuda_enabled = cuda_enabled
        self.device = device

        self.small_decoder = small_decoder

        self.fc1 = nn.Linear(in_dim, l1_dim)
        self.fc2 = nn.Linear(l1_dim, l2_dim)
        self.fc3 = nn.Linear(l2_dim, out_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, target, is_training=True):

        bs = x.size(0)

        if self.small_decoder:
            vj = mask(x, target, is_training,
                      self.cuda_enabled, self.device, small_decoder=self.small_decoder)
        else:
            masked_caps = mask(x, target, is_training,
                               self.cuda_enabled, self.device, small_decoder=self.small_decoder)
            vj = masked_caps.view(bs, -1)

        fc1_out = self.relu(self.fc1(vj))
        fc2_out = self.relu(self.fc2(fc1_out))
        fc3_out = self.sigmoid(self.fc3(fc2_out))

        return fc3_out
