from torch import nn
from torch.nn import functional as F
import torch


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, ignore_index=-100, weight=None, from_probs=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.weight = weight
        self.from_probs = from_probs

    def forward(self, input, target):

        if self.from_probs:
            log_probs = torch.log(input)
            probs = input
        else:
            log_probs = F.log_softmax(input, dim=1)
            probs = torch.exp(log_probs)

        focal_logits = torch.pow(1 - probs, self.gamma) * log_probs

        return F.nll_loss(focal_logits, target, ignore_index=self.ignore_index,
                          weight=self.weight)


import numpy as np
if __name__ == "__main__":
    inp = torch.tensor([ [0.2, 0.8], [0.3, 0.7] ]).float()
    lbl = torch.tensor([0, 1]).long()

    loss_func = FocalLoss(from_probs=True, gamma=0)

    assert loss_func(inp, lbl) == -(np.log(0.2) + np.log(0.7))/2


