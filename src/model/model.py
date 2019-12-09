import math
import pathlib
import sys

import torch
import torch.nn as nn

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent # main directory, the parent of src
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from src.model.ConvLayer import ConvLayer
from src.model.PrimaryCaps import PrimaryCaps
from src.model.DigitCaps import DigitCaps
from src.model.Decoder import Decoder

INPUT_WIDTH = 28
NUM_CONV_IN_CHANNELS = 1
CONV_KERNEL = 9
CONV_STRIDE = 1
NUM_CONV_OUT_CHANNELS = 256
NUM_PRIMARY_CHANNELS = 32
PRIMARY_CAPS_DIM = 8
PRIMARY_KERNEL = 9
PRIMARY_STRIDE = 2
DIGIT_CAPS_DIM = 16
NUM_CLASSES = 10
REGULARIZATION_SCALE = 0.0005
ITER = 3
DEC1_DIM = 512
DEC2_DIM = 1024
CUDA_ENABLED = True
SMALL_DECODER = False
DEVICE = 'cuda:0'
CONV_SHARED_WEIGHTS = 0  # disabled
PRIMARY_SHARED_WEIGHTS = 0  # disabled
DIGIT_SHARED_WEIGHTS = 0  # disabled
CONV_SHARED_BIAS = CONV_SHARED_WEIGHTS  # to have coherency as default
SQUASH_APPROX = False

class Net(nn.Module):

    def __init__(self,
                 input_wh=INPUT_WIDTH,
                 num_conv_in_channels=NUM_CONV_IN_CHANNELS,
                 conv_kernel=CONV_KERNEL,
                 conv_stride=CONV_STRIDE,
                 num_conv_out_channels=NUM_CONV_OUT_CHANNELS,
                 num_primary_channels=NUM_PRIMARY_CHANNELS,
                 primary_caps_dim=PRIMARY_CAPS_DIM,
                 primary_kernel=PRIMARY_KERNEL,
                 primary_stride=PRIMARY_STRIDE,
                 digit_caps_dim=DIGIT_CAPS_DIM,
                 num_classes=NUM_CLASSES,
                 regularization_scale=REGULARIZATION_SCALE,
                 iter=ITER,
                 dec1_dim=DEC1_DIM,
                 dec2_dim=DEC2_DIM,
                 cuda_enabled=CUDA_ENABLED,
                 small_decoder=SMALL_DECODER,
                 device=DEVICE,
                 conv_shared_weights=CONV_SHARED_WEIGHTS,
                 primary_shared_weights=PRIMARY_SHARED_WEIGHTS,
                 digit_shared_weights=DIGIT_SHARED_WEIGHTS,
                 conv_shared_bias=CONV_SHARED_BIAS,
                 squash_approx=SQUASH_APPROX):

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
