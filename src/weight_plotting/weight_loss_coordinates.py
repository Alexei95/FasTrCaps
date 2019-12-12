import argparse
import collections
import itertools
import timeit
import os
import pathlib
import pprint
import sys

import numpy
import torch

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent # main directory, the parent of src
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

DEFAULT_METRIC = torch.nn.MSELoss(reduction='sum')
DEFAULT_MARGIN = 0.05
DEFAULT_NEGATIVE_MARGIN = -0.05
DEFAULT_RANDOM_DIRECTIONS = True
DEFAULT_ORTHOGONAL_DIRECTIONS = False
DEFAULT_NORMALIZE_DIRECTIONS = True
NUM_POINTS = 100

def compute_weight_metric(weights, target=None, metric=DEFAULT_METRIC):
    if target is None:
        target = torch.zeros(weights.size())
    return metric(weights, target)

def compute_model_metric(parameters, targets=None, metric=DEFAULT_METRIC):
    if targets is None:
        targets = [None] * len(list(parameters))
        final_target = None
    else:
        final_target = targets
    return compute_weight_metric(torch.tensor(list(compute_weight_metric(w, t, metric) for w, t in zip(weights, targets))), final_target, metric)

# can be improved by adding options for normalization, filter-wide, layer-wide
# and randomness
def generate_directions(weights, n_dimensions):
    directions = collections.OrderedDict((k, (torch.randn_like(w) for x in range(n_dimensions))) for k, w in weights.items())
    return directions

def compute_landscape_x(n_dimensions, margin=DEFAULT_MARGIN):
    linspace = numpy.linspace(-margin,  margin, num=NUM_POINTS)
    meshes = numpy.meshgrid(*[linspace] * n_dimensions, indexing='ij')
    return meshes

def compute_loss(test_loader, model, loss, directions, coefficients):
    state_dict_backup = model.state_dict()
    parameters = model.state_dict()
    deltas = collections.OrderedDict((k, sum((coefficients[i] * d[i] for i in range(len(d))), 0)) for k, d in directions.items())
    updated_parameters = collections.OrderedDict((k, w + deltas[k]) for k, w in parameters.items())
    model.load_state_dict(updated_parameters)
    avg_loss = sum((loss(model(el)) for el in test_loader), 0) / len(test_loader)
    return avg_loss

def compute_landscape_y(test_loader, model, loss, directions, meshes):
    # itertools.product to iterate over indexes
    # compute loss and save them
    n_dimensions = len(meshes)
    dimensions = meshes[0].shape if n_dimensions > 0 else tuple()
    losses = numpy.zeros(dimensions)
    indexes = itertools.product(*[range(dim) for dim in dimensions])
    for idx in indexes:
        coeffs = tuple(meshes[i][idx] for i in range(n_dimensions))
        l = compute_loss(test_loader, model, loss, directions, coeffs)
        losses.__setitem__(idx, l)
    return losses

# to handle everything, create a class per each model which does the loading
# create base class which raises NotImplementedError or safe_exec to return
# baseline values
# CapsNetLoader.load_model(pickle_file) --> for internal initialization
# CapsNetLoader.run_test_epoch() --> return the average loss over the whole epoch
# CapsNetLoader.weights/parameters --> return the parameters from the model
# CapsNetLoader.update_parameters(parameters) --> loads new parameters
# Alternative: CapsNetLoader.run_test_epoch(parameters) --> uses new parameters for a test epoch


# this class is needed to model a basic nn model to instantiate
# it provides a reference for the needed methods
class MockupModel(object):
    __safe_exec = None
    def __init__(self, safe_exec=False, *args, **kwargs):
        self.__safe_exec = safe_exec

    def __call__(self, *args, **kwargs):
        if not self.__safe_exec:
            raise NotImplementedError
        else:
            return None

# this class is needed to model a basic loss function to instantiate
# it provides a reference for the needed methods
class MockupLoss(object):
    __safe_exec = None
    def __init__(self, safe_exec=False, *args, **kwargs):
        self.__safe_exec = safe_exec

    def __call__(self, *args, **kwargs):
        if not self.__safe_exec:
            raise NotImplementedError
        else:
            return None


class BaseLoader(object):
    __safe_exec = None
    __model = MockupModel()
    __loss = MockupLoss()
    def __init__(self, safe_exec=False, *args, **kwargs):
        self.__safe_exec = safe_exec

    @classmethod
    def load_from_pickle(cls, *args, **kwargs):
        if not cls.__safe_exec:
            raise NotImplementedError
        else:
            return None

def main():
    pass

if __name__ == '__main__':
    main()
