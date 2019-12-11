from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import transforms


class LeNet(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """

    def __init__(self):
        super().__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img, labels=None, is_training=None):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output, torch.zeros(img.size())

    @staticmethod
    def losses(output, reconstruction, target, data, reg_scale, device):
        target = target.to(dtype=torch.long)
        return nn.CrossEntropyLoss()(output, target), torch.Tensor([0]), torch.Tensor([0])

    @staticmethod
    def accuracy(output, labels, cuda):
        output = output.data.max(1, keepdim=True)[1]
        correct_pred = torch.eq(output, labels.view_as(output))  # tensor
        # correct_pred_sum = correct_pred.sum() # scalar. e.g: 6 correct out of 128 images.
        acc = correct_pred.float().mean()  # e.g: 6 / 128 = 0.046875
        return acc

    @staticmethod
    def correct_predictions(output, labels, cuda):
        res = output.data.max(1, keepdim=True)[1]
        return torch.eq(res, labels.view_as(res)).sum()

    @property
    def transforms(self):
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()])
