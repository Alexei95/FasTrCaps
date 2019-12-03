import argparse
import timeit
import os
import pathlib
import pprint
import sys

import torch
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from tqdm import tqdm
import torch.autograd.profiler

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent # main directory, the parent of src
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
