import math
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent # main directory, the parent of src
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from src.model.functions import squash, mask

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

#def squash(input, dim=2):
#    norm = torch.norm(input, dim=dim, keepdim=True)
#    return input * norm / (1 + norm**2)

def update_routing(votes, logits, iterations, dimensions, bias, squash_approx=False):
    if dimensions == 4:     # bs, ci, co, no
        bs, ci, co, no = votes.size()
        votes_trans = votes.permute(3, 0, 1, 2).contiguous()  #no, bs, ci, co
    else:
        bs, ci, co, no, ho, wo = votes.size()
        votes_trans = votes.permute(3, 0, 1, 2, 4, 5).contiguous() # no, bs, ci, co, ho, wo

    for iter in range(iterations):
        route = F.softmax(logits, dim=2)
        preactivate_unrolled = route * votes_trans
        if dimensions == 4:
            preactivate_trans = preactivate_unrolled.permute(1,2,3,0).contiguous()  # bs, ci, co, no
        else:
            preactivate_trans = preactivate_unrolled.permute(1,2,3,0,4,5).contiguous()
                                                                    # bs, ci, co, no, ho, wo
        preactivate = preactivate_trans.sum(dim=1) + bias   #bs, co, no, (ho, wo)
        activation = squash(preactivate, dim=2, squash_approx=squash_approx)             #bs, co, no, (ho, wo)

        act_3d = activation.unsqueeze(1)        #bs, 1, co, no, (ho,wo)
        distances = (votes * act_3d).sum(dim=3) #bs, ci, co, ho, wo
        logits = logits + distances

    return activation

class conv_slim_capsules(nn.Module):
    def __init__(self, ci, ni, co, no, kernel_size, stride, padding, shared_weights=256, cuda_enabled=True, device='cuda:0', squash_approx=False, cuda_enable=True, iter_=1):
        super(conv_slim_capsules, self).__init__()

        self.iterations = iter_

        self.ci = ci
        self.ni = ni
        self.co = co
        self.no = no

        self.shared_weights = shared_weights if shared_weights else 1
        self.squash_approx = squash_approx

        self.conv3d = nn.Conv2d(in_channels = ni,
                                out_channels=int(
                                    co*no / self.shared_weights),
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                bias = False)

        if cuda_enabled:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        self.bias = torch.nn.Parameter(torch.zeros(co, no, 1, 1, device=self.device))

    def forward(self, input):
        bs, ci, ni, hi, wi = input.size()
        input_reshaped = input.view(bs*ci, ni, hi, wi)
        votes = self.conv3d(input_reshaped)  # bs*ci, co*no, ho, wo
        votes = votes.repeat(1, self.shared_weights, 1, 1)
        _, _, ho, wo = votes.size()
        votes_reshaped = votes.view(bs, ci, self.co, self.no, ho, wo).contiguous()
        logits = votes_reshaped.new(bs,ci,self.co,ho,wo).zero_()

        activation = update_routing(votes_reshaped, logits, self.iterations, 6, self.bias, squash_approx=self.squash_approx)

        return activation

class capsule(nn.Module):
    def __init__(self, ci, ni, co, no, iter_, cuda_enabled=True, device='cuda:0', shared_weights=36, squash_approx=False):
        super(capsule, self).__init__()

        if cuda_enabled:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        self.shared_weights = shared_weights if shared_weights else 1
        self.squash_approx = squash_approx

        self.weight = nn.Parameter(torch.randn(int(ci / self.shared_weights), ni, co*no))
        self.bias = nn.Parameter(torch.zeros(co, no))
        self.ci = ci
        self.co = co
        self.no = no

        self.iter_ = iter_

    def forward(self, input):
        bs = input.size(0)
        weights = self.weight.repeat(self.shared_weights, 1, 1)
        votes = (input.unsqueeze(3) * weights).sum(dim=2).view(-1, self.ci, self.co, self.no)   #bs, ci, co*no
        logits = votes.new(bs, self.ci, self.co).zero_()
        activation = update_routing(votes, logits, self.iter_, 4, self.bias, squash_approx=self.squash_approx)

        return activation


class decoder(nn.Module):

    def __init__(self, in_dim, l1_dim, l2_dim, out_dim, cuda_enabled=True, device='cuda:0', small_decoder=False):

        super(decoder, self).__init__()

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

class CapsNet(nn.Module):
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
        super(CapsNet, self).__init__()

        self.cuda_enabled = cuda_enabled
        if cuda_enabled:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        conv_dimension = math.floor(
            (input_wh-conv_kernel+conv_stride)/conv_stride)
        primary_dimension = math.floor(
            (conv_dimension-primary_kernel+primary_stride)/primary_stride)

        self.conv_shared_weights = conv_shared_weights if conv_shared_weights else 1
        self.conv_shared_bias = conv_shared_bias if conv_shared_bias else 1
        self.conv = nn.Conv2d(in_channels = num_conv_in_channels,
                                out_channels = int(num_conv_out_channels / self.conv_shared_weights),
                                kernel_size = conv_kernel,
                                stride = conv_stride)
        self.relu = nn.ReLU()
        self.primary = conv_slim_capsules(ci=1, ni=num_conv_out_channels, co=num_primary_channels, no=primary_caps_dim,
                                          kernel_size=primary_kernel, stride=primary_stride, padding=0,
                                          device=self.device,
                                          cuda_enabled=cuda_enabled,
                                          shared_weights=primary_shared_weights,
                                          squash_approx=squash_approx)
        self.digit = capsule(ci=num_primary_channels*primary_dimension*primary_dimension,
        ni=primary_caps_dim, co=num_classes, no=digit_caps_dim,
                             device=device,
                             cuda_enabled=cuda_enabled,
                             iter_=iter,
                             shared_weights=digit_shared_weights,
                             squash_approx=squash_approx)
        decoder_in_dim = digit_caps_dim if small_decoder else num_classes * digit_caps_dim
        self.decoder = decoder(in_dim=decoder_in_dim, l1_dim=dec1_dim,
                                    l2_dim=dec2_dim, out_dim=input_wh**2, device=self.device,
                                    small_decoder=small_decoder)

    def forward(self, input, labels, is_training=True):
        out_conv = self.conv(input)
        out_conv = out_conv.repeat(1, self.conv_shared_weights, 1, 1)
        out_conv = self.relu(out_conv.unsqueeze(1))

        out_primary = self.primary(out_conv)
        bs, c, n, h, w = out_primary.size()
        out_primary = out_primary.permute(0,1,3,4,2).contiguous().view(bs,-1,n)


        out_digit = self.digit(out_primary)

        reconstruction = self.decoder(out_digit, labels, is_training)

        return out_digit, reconstruction

def margin_loss(out_digit, labels, device='cuda:0'):
    # labels  onehot
    margin = 0.4
    logits = torch.norm(out_digit, dim=2) - 0.5
    positive_cost = labels * torch.lt(logits, margin).float() * torch.pow(logits - margin, 2)
    negative_cost = (1-labels) * torch.gt(logits, -margin).float() * torch.pow(logits + margin, 2)
    return 0.5 * positive_cost + 0.5 * 0.5 * negative_cost

def loss_func(out_digit, labels, scale, reconstruction, image, device='cuda:0'):
    mloss = margin_loss(out_digit, labels)
    rloss = (reconstruction-image)**2

    mloss = mloss.sum(dim=1).mean()
    rloss = rloss.sum(dim=1).mean()
    tloss = mloss + scale * rloss

    return tloss, mloss, rloss
