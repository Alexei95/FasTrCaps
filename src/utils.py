"""Utilities

PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829

Author: Cedric Chee
"""
import argparse
import io
import math
import os
import os.path
import pickle
import pathlib
import sys

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
try:
    import dill
    import scipy
except ImportError:
    dill = None
import numpy


PROJECT_DIR = pathlib.Path(__file__).absolute().resolve().parent.parent # main directory, the parent of src
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))


TIME_FORMAT = '%Y_%m_%d_%H_%M_%S_%z'
RESULT_DIRECTORY = PROJECT_DIR / 'results'
PICKLE_COMPRESSION = True
DEFAULT_PICKLE_EXTENSION = 'pkl'
SAVE_FULL_OBJECT = True
COMPRESSIONS = ('lzma', 'zip', 'tar.gz', 'tar.bz2')
DEFAULT_PICKLE_COMPRESSION = COMPRESSIONS[0]  # lzma
DILL_DUMPED_NONE = b'\x80\x03N.'

# wrapper to use compress pickle if installed
def dump(obj, filename, default_compression=DEFAULT_PICKLE_COMPRESSION, directory=RESULT_DIRECTORY, use_compression=PICKLE_COMPRESSION, save_full_object=SAVE_FULL_OBJECT):
    filename = str(filename)
    if not os.path.isabs(filename):
        filename = os.path.join(directory, filename)

    try:
        os.makedirs(os.path.dirname(filename))
    except OSError:
        pass

    if dill is not None and save_full_object:
        string_buff = dill.dumps(obj, recurse=True)
        obj_to_save = {'dill': True, 'obj': string_buff}
    else:
        obj_to_save = obj

    if compress_pickle is not None and use_compression:
        if os.path.splitext(filename)[0] == filename:
            filename = '.'.join([filename, default_compression])
        compress_pickle.dump(obj_to_save, filename)
    else:
        if os.path.splitext(filename)[0] == filename:
            filename = '.'.join([filename, use_compression])
        with open(filename, 'wb') as f:
            pickle.dump(obj_to_save, f)

    return filename


def load(pickle_file=None, directory=RESULT_DIRECTORY, default_compression=DEFAULT_PICKLE_COMPRESSION, use_compression=PICKLE_COMPRESSION, save_full_object=SAVE_FULL_OBJECT, none=DILL_DUMPED_NONE):
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

    compressed = filename.lower().endswith(COMPRESSIONS)

    if compress_pickle is not None and (compressed or use_compression):
        if os.path.splitext(filename)[0] == filename:
            fname = '.'.join([filename, default_compression])
        res = compress_pickle.load(filename, encoding='latin1')
    else:
        if os.path.splitext(filename)[0] == filename:
            fname = '.'.join([filename, DEFAULT_PICKLE_EXTENSION])
        with open(filename, 'rb') as f:
            res = pickle.load(f, encoding='latin1')

    if dill is not None and isinstance(res, dict) and res.get('dill', None):
        string_buff = res.get('obj', none)
        obj = dill.loads(string_buff, ignore=True)
    else:
        obj = res

    return obj


def one_hot_encode(target, length):
    """Converts batches of class indices to classes of one-hot vectors."""
    batch_s = target.size(0)
    one_hot_vec = torch.zeros(batch_s, length)

    for i in range(batch_s):
        one_hot_vec[i, target[i]] = 1.0

    return one_hot_vec

def make_full_checkpoint_obj(dict_=locals(), dict_globals=globals()):
    model = dict_.get('model', None)
    optimizer = dict_.get('optimizer', None)
    lr_wr = dict_.get('lr_wr', None)
    return {
            'epoch': dict_.get('epoch', float('nan')) + 1,
            'model': model,
            'optimizer': optimizer,
            'model_state_dict': model.state_dict() if model is not None else None,
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'lr_wr' : lr_wr.__dict__ if lr_wr is not None else None,
            'lr_wr_obj': lr_wr,
            'args': dict_.get('args', None),
            'loss_func': dict_globals.get('loss_func', None),
            'train_loader': dict_.get('train_loader', None),
            'test_loader': dict_.get('test_loader', None),
            'one_hot_encode': dict_globals['utils'].one_hot_encode if 'utils' in dict_globals else None,
            }

def make_partial_checkpoint_obj(dict_=locals(), dict_globals=globals()):
    model = dict_.get('model', None)
    optimizer = dict_.get('optimizer', None)
    lr_wr = dict_.get('lr_wr', None)
    return {
            'epoch': dict_.get('epoch', float('nan')) + 1,
            'model': model,
            'optimizer': optimizer,
            'model_state_dict': model.state_dict() if model is not None else None,
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'lr_wr' : lr_wr.__dict__ if lr_wr is not None else None,
            'lr_wr_obj': lr_wr,
            'args': dict_.get('args', None),
            'loss_func': dict_globals.get('loss_func', None),
            'train_loader': dict_.get('train_loader', None),
            'test_loader': dict_.get('test_loader', None),
            'one_hot_encode': dict_globals['utils'].one_hot_encode if 'utils' in dict_globals else None,
            }

def make_dataset_obj(dict_=locals(), dict_globals=globals()):
    return {
            'args': dict_.get('args', None),
            'train_loader': dict_.get('train_loader', None),
            'test_loader': dict_.get('test_loader', None),
            }


def checkpoint(state, epoch, directory):
    """Save checkpoint"""
    model_out_path = pathlib.Path(directory) / 'trained_model' / 'model_epoch_{}'.format(epoch)
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

    vutils.save_image(image_tensor, str(file_name))


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
