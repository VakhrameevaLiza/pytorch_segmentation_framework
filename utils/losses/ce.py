from torch import nn
from torch.nn import functional as F
import torch
import numpy as np

class CELoss(nn.Module):

    def __init__(self, weight=None):
        super(CELoss, self).__init__()
        self.weight = weight

    def forward(self, input, target, alpha=None, rolled=None):

        if alpha is None and rolled is None:
            return nn.CrossEntropyLoss(weight=self.weight)(input, target)

        bs, cl, h, w = input.shape
        target_ohe = torch.FloatTensor(bs, cl, h, w).cuda()
        target_ohe.zero_()
        target_ohe.scatter_(1, target[:, None, :, :], 1)

        if alpha is not None and rolled is not None:
           target_ohe = alpha* target_ohe + (1-alpha) * target_ohe[rolled, ...]

        log_softmax = torch.nn.LogSoftmax(dim=1)
        log_probs = log_softmax(input)

        nll = - log_probs * target_ohe
        nll = nll.mean(dim=(2,3))
        if self.weight is not None:
            nll *= self.weight
        nll = nll.sum(dim=1).mean()

        return nll
