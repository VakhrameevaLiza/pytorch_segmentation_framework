from torch import nn
from torch.nn import functional as F
import torch


class BCELoss(nn.Module):

    def __init__(self, weight=None):
        super(BCELoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        if self.weight is not None:
            weight = self.weight[target]
        else:
            weight = None
        return F.binary_cross_entropy(torch.sigmoid(input), target.float(), weight=weight)
