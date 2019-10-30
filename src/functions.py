import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy
import math


def squash_v3_bea(x, dim=3):
    eps = 1e-8
    norm_x = norm(x, dim=dim)
    norm_sq_x = norm_x * norm_x

    return (norm_sq_x / (1.0 + norm_sq_x)) * (x / (norm_x + eps))


def norm(x, dim=2):
    norm_sq_x = torch.sum(x**2, dim=dim, keepdim=True)
    norm_x = torch.sqrt(norm_sq_x)

    return norm_x


def reluPrime(x):
    y = x.clone()
    y[y <= 0] = 0
    y[y > 0] = 1
    return y


def sigmoidPrime(x):
    sigmoid = nn.Sigmoid()
    return sigmoid(x) * (1-sigmoid(x))


def squash_v1(x, dim=3):
    norm_sq_x = torch.sum(x**2, dim=dim, keepdim=True)

    norm_x = torch.sqrt(norm_sq_x)

    return (norm_sq_x / (1.0 + norm_sq_x)) * (x / norm_x)


def squash_v2_bea(x, dim=3):
    norm = x.norm(dim=dim, keepdim=True)

    return torch.where(norm <= 2, x / 2, x / norm)


def squash(x, dim=3, squash_approx=False):
    if squash_approx:
        return squash_v2_bea(x, dim=dim)
    else:
        return squash_v3_bea(x, dim=dim)


def squashPrime(s, dim=3):
    # return dv / ds  ,  v = squash(s)
    # v = bs  1  10  16
    # s = bs  1  10  16
    # dv/ds = bs  1  10  16  16

    norm_sq_s = torch.sum(s**2, dim=dim, keepdim=True)
    #  bs  1  10  1
    norm_s = torch.sqrt(norm_sq_s)
    #  bs  1  10  1

    A = (1.0 - norm_sq_s) / (norm_s * (1.0 + norm_sq_s)**2)
    B = norm_s / (1.0 + norm_sq_s)
    #  bs  1  10  1

    #
    one = torch.ones_like(s.data)
    diag = B * one  # bs 1 10 16
    diag = torch.diag_embed(diag, dim1=dim, dim2=(dim+1))   # bs 1 10 16  16

    dvds = torch.matmul((s*A).unsqueeze(dim+1),
                        s.unsqueeze(dim))  # bs  1  10  16  16
    dvds = dvds + diag

    return dvds    # bs 1 10 16 16


def softmaxPrime(b, dim=2):
    #  b :  bs  1152  10  1
    #  c :  bs  1152  10  1
    # dc/db: bs  1152  10  10

    b1 = F.softmax(b, dim=dim)  # bs  1152  10  1

    dcdb = torch.matmul(b1, b1.transpose(dim, dim+1))   # bs  1152  10  10

    b1 = torch.diag_embed(b1.squeeze(dim+1), dim1=dim,
                          dim2=dim+1)   # bs  1152  10  10

    dcdb = b1 - dcdb

    return dcdb  # bs  1152  10  10


def mask(out_digit_caps, target, is_training=True, cuda_enabled=True, device='cuda:0', small_decoder=False):
    """

    # Returns:
        # masked: [batch_size, 10, 16] The masked capsules tensors.
    # """
    batch_index = torch.arange(out_digit_caps.size(0))
    device = torch.device(device)
    if small_decoder:
        masked_out = torch.zeros(out_digit_caps.size(
            0), out_digit_caps.size(2), device=device)
    else:
        masked_out = torch.zeros(out_digit_caps.size(), device=device)

    if is_training:
        if small_decoder:
            masked_out[batch_index, :] = out_digit_caps[batch_index, target, :]
        else:
            masked_out[batch_index, target,
                       :] = out_digit_caps[batch_index, target, :]

    else:
        # # a) Get capsule outputs lengths, ||v_c||
        length = (norm(out_digit_caps, dim=2)).squeeze(2)

        # # b) Pick out the index of longest capsule output, v_length by
        # # masking the tensor by the max value in dim=1.
        _, max_index = length.max(dim=1)
        if small_decoder:
            masked_out[batch_index,
                       :] = out_digit_caps[batch_index, max_index, :]
        else:
            masked_out[batch_index, max_index,
                       :] = out_digit_caps[batch_index, max_index, :]

    return masked_out


def loss(out_digit, reconstruction, target, image, regularization_scale, device='cuda:0'):
    # margin loss

    bs = out_digit.size(0)

    # v_c = torch.sqrt((out_digit**2).sum(dim=2))
    v_c = (norm(out_digit, dim=2)).squeeze(2)

    m_p = 0.9
    m_m = 0.1
    lmbd = 0.5

    zero = torch.zeros(1, device=torch.device(device))

    m1 = (torch.max(zero, m_p-v_c))**2
    m0 = (torch.max(zero, v_c-m_m))**2
    # m1 = (1.0/2 + 1.0/4 ) * (1. - v_c)**2
    # m0 = (1.0/4 + 1.0/8 ) * v_c**2

    # m1 = 0.875 * (1.0 - v_c**2)
    # m0 = 0.4375 * v_c**2

    Lk = target * m1 + lmbd * (1.0 - target) * m0

    MarginL = Lk.sum(dim=1).mean(0)

    # reconstruction loss

    image = image.view(bs, -1)

    error = 2 * (reconstruction - image)**2
    ReconstructionL = error.sum(dim=1).mean(0)

    TotalL = MarginL + regularization_scale * ReconstructionL

    return TotalL, MarginL, ReconstructionL
