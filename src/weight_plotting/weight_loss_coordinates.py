import argparse
import timeit
import os
import pathlib
import pprint
import sys

import torch

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent # main directory, the parent of src
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

DEFAULT_METRIC = torch.nn.MSELoss(reduction='sum')
DEFAULT_POSITIVE_MARGIN = 0.05
DEFAULT_NEGATIVE_MARGIN = -0.05

def compute_weight_metric(weights, target=None, metric=DEFAULT_METRIC):
    if target is None:
        target = torch.zeros(weights.size())
    return metric(weights, target)

def compute_model_metric(parameters, targets=None, metric=DEFAULT_METRIC):
    if targets is None:
        targets = [None] * len(list(weights))
        final_target = None
    else:
        final_target = targets
    return compute_weight_metric(torch.tensor(list(compute_weight_metric(w, t, metric) for w, t in zip(weights, targets))), final_target, metric)

def compute_landscape_x(parameters, pos_margin=DEFAULT_POS_MARGIN):
    pass


