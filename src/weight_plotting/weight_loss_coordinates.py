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

import src.utils as utils

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


DEFAULT_REGULARIZATION_SCALE = 0.0005
DEFAULT_USE_RECONSTRUCTION_LOSS = True
DEFAULT_NUM_CLASSES = 10


class CapsNetLoader(object):
    __model = None
    __test_loader = None
    __loss_func = None
    __backup_parameters = None
    __one_hot_encode = None
    __use_reconstruction_loss = None
    __num_classes = None
    __device = None
    __regularization_scale = None

    def load_model(self, pickle_file):
        obj = utils.load(pickle_file)
        args = obj['args']
        self.__model = obj['model']
        self.__test_loader = obj['test_loader']
        self.__loss_func = obj['loss_func']
        if args is not None:
            self.__regularization_scale = args.regularization_scale
            self.__use_reconstruction_loss = args.use_reconstruction_loss
            self.__num_classes = args.num_classes
        else:
            self.__regularization_scale = DEFAULT_REGULARIZATION_SCALE
            self.__use_reconstruction_loss = DEFAULT_USE_RECONSTRUCTION_LOSS
            self.__num_classes = DEFAULT_NUM_CLASSES
        self.__one_hot_encode = obj['one_hot_encode']

    def save_backup_parameters(self):
        self.__backup_parameters = self.__model.parameters()

    def load_backup_parameters(self):
        self.__model.load_state_dict(self.__backup_parameters)

    @property
    def weights(self):
        return list(self.__model.parameters())

    def update_parameters(self, parameters):
        self.__model.load_state_dict(parameters)

    @staticmethod
    def run_capsnet_test_batch(model, loss_func, one_hot_encode, num_classes, data, target, device, regularization_scale, use_reconstruction_loss):
        with torch.no_grad():
            batch_size = data.size(0)
            target_indices = target
            target_one_hot = one_hot_encode(target_indices, length=num_classes)
            assert target_one_hot.size() == torch.Size([batch_size, 10])

            target = target_one_hot

            if device:
                data = data.to(device)
                target = target.to(device)
                target_indices.to(device)

            # Output predictions
            output, reconstruction = model(data, target_indices, False) # output from DigitCaps (out_digit_caps)

            # Sum up batch loss
            t_loss, m_loss, r_loss = loss_func(
                output, target, regularization_scale, reconstruction, data, device, batch_size)
            loss = t_loss.data
            margin_loss = m_loss.data
            recon_loss = r_loss.data

            # Count number of correct predictions
            # v_magnitude shape: [128, 10, 1, 1]
            v_magnitude = torch.sqrt((output**2).sum(dim=2, keepdim=True))
            # pred shape: [128, 1, 1, 1]
            pred = v_magnitude.data.max(1, keepdim=True)[1].cpu()
            correct = pred.eq(target_indices.view_as(pred)).sum()

            return {'loss': loss,
                    'margin_loss': margin_loss,
                    'recon_loss': recon_loss,
                    'correct_predictions': correct,
                    'batch_size': batch_size}

    def _pre_run_hook(self, parameters=None):
        # if parameters are None uses the current parameters
        # otherwise it backups the old ones, loads the new ones, runs the test
        # and reloads the old ones
        if parameters is not None:
            self.save_backup_parameters()
            self.update_parameters(parameters)
        # Switch to evaluate mode
        self.__model.eval()

    def _post_run_hook(self, parameters=None):
        if parameters is not None:
            self.load_backup_parameters()

    def run_test_batch(self, parameters=None):
        self._pre_run_hook(parameters=parameters)

        res = self.run_capsnet_test_batch(self.__model, self.__loss_func,
                                              self.__one_hot_encode,
                                              self.__num_classes, data, target,
                                              self.__device,
                                              self.__regularization_scale,
                                              self.__use_reconstruction_loss)

        self._post_run_hook(paramters=parameters)

        # to check whether the result has a get method
        # it is actually not needed since this method is internal, but still
        # it can be useful in other versions
        if not callable(getattr(res, 'get', None)):
            res = {}

        return {'loss': res.get('loss', float('+inf')),
                'margin_loss': res.get('margin_loss', float('+inf')),
                'recon_loss': res.get('recon_loss', float('+inf')),
                'correct_predictions': res.get('correct_predictions', 0),
                'batch_size': res.get('batch_size', 1)}

    def run_test_epoch(self, parameters=None):
        loss = 0
        margin_loss = 0
        recon_loss = 0

        correct = 0

        num_batches = len(self.__test_loader)

        self._pre_run_hook(parameters=parameters)

        for data, target in self.__test_loader:
            res = self.run_test_batch(parameters=None)
            # results are averaged across the batch
            loss += res.get('loss', float('+inf'))
            margin_loss += res.get('margin_loss', float('+inf'))
            recon_loss += res.get('recon_loss', float('+inf'))
            correct += res.get('correct_predictions', float('+inf'))

        # Average of test losses
        loss /= num_batches
        margin_loss /= num_batches
        recon_loss /= num_batches

        self._post_run_hook(parameters=parameters)

        # Log test accuracies
        # num_test_data = len(self.__test_loader.dataset)
        # accuracy = correct / num_test_data
        # accuracy_percentage = float(correct) * 100.0 / float(num_test_data)

        return {'loss': loss,
                'margin_loss': margin_loss,
                'recon_loss': recon_loss}

# the flow is:
# - generate directions from layer parameters
# - for each point to plot:
# -- compute the amount to move (they will be the coordinates)
# -- compute new parameters as old + direction * move (per layer or depending on config)
# -- compute loss by running full test epoch with new parameters
# - plot all the points


def main():
    pass

if __name__ == '__main__':
    main()
