import torch
import torch.nn as nn

from ConvLayer import ConvLayer
from PrimaryCaps import PrimaryCaps
from DigitCaps import DigitCaps
from Decoder import Decoder

import math


class Net(nn.Module):

    def __init__(self, input_wh, num_conv_in_channels, conv_kernel, conv_stride, num_conv_out_channels, num_primary_channels,
                 primary_caps_dim, primary_kernel, primary_stride, digit_caps_dim, num_classes, regularization_scale, iter, dec1_dim, dec2_dim,
                 cuda_enabled=True, small_decoder=True, device='cuda:0', conv_shared_weights=256, primary_shared_weights=256, digit_shared_weights=36, conv_shared_bias=256, squash_approx=False):

        super(Net, self).__init__()

        self.cuda_enabled = cuda_enabled
        if cuda_enabled:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        self.regularization_scale = regularization_scale

        conv_dimension = math.floor(
            (input_wh-conv_kernel+conv_stride)/conv_stride)
        primary_dimension = math.floor(
            (conv_dimension-primary_kernel+primary_stride)/primary_stride)

        self.conv = ConvLayer(in_channels=num_conv_in_channels,
                              out_channels=num_conv_out_channels,
                              kernel_size=conv_kernel,
                              stride=conv_stride,
                              cuda_enabled=cuda_enabled,
                              device=device,
                              shared_weights=conv_shared_weights,
                              shared_bias=conv_shared_bias)

        self.primary = PrimaryCaps(in_channels=num_conv_out_channels,
                                   out_channels=num_primary_channels,
                                   out_caps_dim=primary_caps_dim,
                                   kernel_size=primary_kernel,
                                   stride=primary_stride,
                                   cuda_enabled=cuda_enabled,
                                   device=device,
                                   shared_weights=primary_shared_weights,
                                   squash_approx=squash_approx)

        self.digit = DigitCaps(in_dim=num_primary_channels*primary_dimension*primary_dimension,
                               out_dim=num_classes,
                               in_caps_dim=primary_caps_dim,
                               out_caps_dim=digit_caps_dim,
                               iter=iter,
                               cuda_enabled=cuda_enabled,
                               device=device,
                               shared_weights=digit_shared_weights,
                               squash_approx=squash_approx)

        decoder_in_dim = digit_caps_dim if small_decoder else num_classes * digit_caps_dim
        self.decoder = Decoder(in_dim=decoder_in_dim,
                               l1_dim=dec1_dim,
                               l2_dim=dec2_dim,
                               out_dim=input_wh*input_wh,
                               device=device,
                               small_decoder=small_decoder)

    def forward(self, x, labels, is_training=True):
        out_conv = self.conv(x)

        out_primary = self.primary(out_conv)

        out_digit = self.digit(out_primary)

        reconstruction = self.decoder(out_digit, labels, is_training)

        return out_digit, reconstruction
