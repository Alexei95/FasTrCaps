"""Utilities

PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829

Author: Cedric Chee
"""
import argparse
import math
import os
import os.path
import pickle
import pathlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets
import torchvision.utils as vutils

try:
    import compress_pickle
except ImportError:
    compress_pickle = None
import numpy


TIME_FORMAT = '%Y_%m_%d_%H_%M_%S_%z'
RESULT_DIRECTORY = pathlib.Path(__file__).absolute().resolve().parent.parent / 'results'
PICKLE_COMPRESSION = True
DEFAULT_PICKLE_COMPRESSION = 'lzma'
DEFAULT_PICKLE_EXTENSION = 'pkl'

# wrapper to use compress pickle if installed
def dump(obj, filename, default_compression=DEFAULT_PICKLE_COMPRESSION, directory=RESULT_DIRECTORY):
    filename = str(filename)
    if not os.path.isabs(filename):
        filename = os.path.join(directory, filename)

    try:
        os.makedirs(os.path.dirname(filename))
    except OSError:
        pass

    if compress_pickle is not None and PICKLE_COMPRESSION:
        if os.path.splitext(filename)[0] == filename:
            filename = '.'.join([filename, default_compression])
        compress_pickle.dump(obj, filename)
    else:
        if os.path.splitext(filename)[0] == filename:
            filename = '.'.join([filename, DEFAULT_PICKLE_EXTENSION])
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

    return filename


def load(pickle_file=None, directory=RESULT_DIRECTORY, default_compression=DEFAULT_PICKLE_COMPRESSION, use_compression=PICKLE_COMPRESSION):
    directory = str(directory)
    if pickle_file is not None:
        pickle_file = str(pickle_file)
        if os.path.isabs(pickle_file):
            filename = pickle_file
        else:
            filename = os.path.join(directory, pickle_file)
    else:
        files = os.listdir(directory)
        filename = os.path.join(directory, sorted(files)[-1])

    compressed = filename.endswith(default_compression)

    if compressed and compress_pickle is not None and use_compression:
        if os.path.splitext(filename)[0] == filename:
            fname = '.'.join([filename, default_compression])
        res = compress_pickle.load(filename, encoding='latin1')
    else:
        if os.path.splitext(filename)[0] == filename:
            fname = '.'.join([filename, DEFAULT_PICKLE_EXTENSION])
        with open(filename, 'rb') as f:
            res = pickle.load(f, encoding='latin1')

    return res


def one_hot_encode(target, length):
    """Converts batches of class indices to classes of one-hot vectors."""
    batch_s = target.size(0)
    one_hot_vec = torch.zeros(batch_s, length)

    for i in range(batch_s):
        one_hot_vec[i, target[i]] = 1.0

    return one_hot_vec


def checkpoint(state, epoch, directory):
    """Save checkpoint"""
    model_out_path = pathlib.Path(directory) / 'trained_model' / 'model_epoch_{}.pth'.format(epoch)
    dump(state, model_out_path)
    print('Checkpoint saved to {}'.format(model_out_path))


def load_mnist(args):
    """Load MNIST dataset.
    The data is split and normalized between train and test sets.
    """
    # Normalize MNIST dataset.
    if args.normalize_input:
        data_transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        data_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        data_transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(
                2/28, 2/28), scale=None, shear=None, resample=False, fillcolor=0),
            transforms.ToTensor()
            # , transforms.Normalize((0.1307,), (0.3081,))
        ])

        data_transform_test = transforms.Compose([
            transforms.ToTensor()
            # , transforms.Normalize((0.1307,), (0.3081,))
        ])

    kwargs = {'num_workers': args.threads,
              'pin_memory': True} if args.cuda else {}

    print('===> Loading MNIST training datasets')
    # MNIST dataset
    training_set = datasets.MNIST(
        str(pathlib.Path(args.data_directory) / 'mnist'), train=True, download=True, transform=data_transform_train)
    # Input pipeline
    training_data_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    print('===> Loading MNIST testing datasets')
    testing_set = datasets.MNIST(
        str(pathlib.Path(args.data_directory) / 'mnist'), train=False, download=True, transform=data_transform_test)
    testing_data_loader = DataLoader(
        testing_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return training_data_loader, testing_data_loader


def load_fashionmnist(args):
    """Load MNIST dataset.
    The data is split and normalized between train and test sets.
    """
    # Normalize MNIST dataset.
    if args.normalize_input:
        data_transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        data_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        data_transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(
                2/28, 2/28), scale=None, shear=None, resample=False, fillcolor=0),
            transforms.ToTensor()
            # , transforms.Normalize((0.1307,), (0.3081,))
        ])

        data_transform_test = transforms.Compose([
            transforms.ToTensor()
            # , transforms.Normalize((0.1307,), (0.3081,))
        ])

    kwargs = {'num_workers': args.threads,
              'pin_memory': True} if args.cuda else {}

    print('===> Loading Fashion MNIST training datasets')
    # MNIST dataset
    training_set = datasets.FashionMNIST(
        str(pathlib.Path(args.data_directory) / 'fashionmnist'), train=True, download=True, transform=data_transform_train)
    # Input pipeline
    training_data_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    print('===> Loading Fashion MNIST testing datasets')
    testing_set = datasets.FashionMNIST(
        str(pathlib.Path(args.data_directory) / 'fashionmnist'), train=False, download=True, transform=data_transform_test)
    testing_data_loader = DataLoader(
        testing_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return training_data_loader, testing_data_loader


def load_cifar10(args):
    """Load CIFAR10 dataset.
    The data is split and normalized between train and test sets.
    """
    # Normalize CIFAR10 dataset.
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    kwargs = {'num_workers': args.threads,
              'pin_memory': True} if args.cuda else {}

    print('===> Loading CIFAR10 training datasets')
    # CIFAR10 dataset
    training_set = datasets.CIFAR10(
        str(pathlib.Path(args.data_directory) / 'cifar10'), train=True, download=True, transform=data_transform)
    # Input pipeline
    training_data_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    print('===> Loading CIFAR10 testing datasets')
    testing_set = datasets.CIFAR10(
        str(pathlib.Path(args.data_directory) / 'cifar10'), train=False, download=True, transform=data_transform)
    testing_data_loader = DataLoader(
        testing_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return training_data_loader, testing_data_loader


def load_data(args):
    """
    Load dataset.
    """
    dst = args.dataset

    if dst == 'mnist':
        return load_mnist(args)
    elif dst == 'fashionmnist':
        return load_fashionmnist(args)
    elif dst == 'cifar10':
        return load_cifar10(args)
    else:
        raise Exception(
            'Invalid dataset, please check the name of dataset:', dst)


def save_image(image, file_name):
    """
    Save a given image into an image file
    """
    # Check number of channels in an image.
    if image.size(1) == 2:
        # 2-channel image
        zeros = torch.zeros(image.size(0), 1, image.size(2), image.size(3))
        image_tensor = torch.cat([zeros, image.data.cpu()], dim=1)
    else:
        # Grayscale or RGB image
        image_tensor = image.data.cpu()  # get Tensor from Variable

    vutils.save_image(image_tensor, file_name)


def accuracy(output, target, cuda_enabled=True):
    """
    Compute accuracy.

    Args:
        output: [batch_size, 10, 16, 1] The output from DigitCaps layer.
        target: [batch_size] Labels for dataset.

    Returns:
        accuracy (float): The accuracy for a batch.
    """
    batch_size = target.size(0)

    v_length = torch.sqrt((output**2).sum(dim=2, keepdim=True))
    softmax_v = F.softmax(v_length, dim=1)
    assert softmax_v.size() == torch.Size([batch_size, 10, 1])

    _, max_index = softmax_v.max(dim=1)
    assert max_index.size() == torch.Size([batch_size, 1])

    pred = max_index.view(batch_size)  # max_index.squeeze() #
    assert pred.size() == torch.Size([batch_size])

    if cuda_enabled:
        target = target.cuda()
        pred = pred.cuda()

    correct_pred = torch.eq(target, pred.data)  # tensor
    # correct_pred_sum = correct_pred.sum() # scalar. e.g: 6 correct out of 128 images.
    acc = correct_pred.float().mean()  # e.g: 6 / 128 = 0.046875

    return acc


def to_np(param):
    """
    Convert values of the model parameters to numpy.array.
    """
    return param.clone().cpu().data.numpy()


def str2bool(v):
    """
    Parsing boolean values with argparse.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def exponential_decay_LRR(optimizer, lr, global_step, decay_steps, decay_rate, staircase=True):
    if staircase:
        lr_new = lr * (decay_rate ** (global_step // decay_steps))
    else:
        lr_new = lr * (decay_rate ** (global_step / decay_steps))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new


def find_lr(optimizer, step):
    lr_new = 10**(step/100 - 6)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new


def one_cycle_policy(optimizer, start_lr, global_step, total_steps):
    tsoc = 0.9 * total_steps

    if global_step <= 0.9 * total_steps / 2:
        lr_new = 9 * start_lr * global_step / (tsoc/2) + start_lr
    elif global_step > 0.9 * total_steps:
        lr_new = 0.9 * start_lr * \
            (global_step - total_steps) / (tsoc - total_steps) + start_lr / 10
    else:
        lr_new = -9 * start_lr * (global_step - tsoc/2) / \
            (tsoc/2) + 10 * start_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new


class warm_restarts():
    def __init__(self, Ti, Tmult, emin, emax):
        super(warm_restarts, self).__init__()

        self.Ti = Ti
        self.Tmult = Tmult
        self.Tcur = 0
        self.emin = emin
        self.emax = emax

    def update_lr(self, optimizer, num_batches):
        e = self.emin + 0.5 * (self.emax - self.emin) * (1 +
                                                         math.cos(math.pi * self.Tcur / (self.Ti*num_batches)))

        if self.Tcur == self.Ti*num_batches:
            self.Tcur = 0
            self.Ti = self.Ti * self.Tmult
        else:
            self.Tcur = self.Tcur + 1

        for param_group in optimizer.param_groups:
            param_group['lr'] = e


class custom_warm_restarts():
    def __init__(self, emin, emax):
        super(custom_warm_restarts, self).__init__()

        self.Ti = 60000 * 3
        self.Tcur = 0
        self.emin = emin
        self.emax = emax

    def update_lr(self, optimizer):
        e = self.emin + 0.5 * (self.emax - self.emin) * \
            (1 + math.cos(math.pi * self.Tcur / self.Ti))

        if (self.Tcur == 60000*3) and (self.Ti == 60000*3):
            self.Tcur = 0
            self.Ti = 97 * 3750
        elif self.Tcur == self.Ti:
            self.Tcur = 0
        else:
            self.Tcur = self.Tcur + 1

        for param_group in optimizer.param_groups:
            param_group['lr'] = e


def write_tensor(x, path):
    for i in range(x.size(0)):
        for j in range(x.size(1)):
            for k in range(x.size(2)):
                path.write('%.10f \n' % x[i, j, k].item())
