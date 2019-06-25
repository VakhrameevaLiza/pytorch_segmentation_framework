import torch.nn as nn
import numpy as np
from utils.losses.focal import FocalLoss
from utils.losses.dice import DiceLoss
from utils.losses.bce import BCELoss
from utils.losses.ce import CELoss
from utils.losses.ohem import ProbOhemCrossEntropy2d
import torch


def get_loss_func(loss_types, loss_coefs,
                          num_cat=0, num_attr=0,
                          cat_weights=None, attr_weights=None):

    assert len(loss_types) == len(loss_coefs)

    if cat_weights is not None:
        cat_weights = torch.from_numpy(cat_weights).float()
        if torch.cuda.is_available():
            cat_weights = cat_weights.cuda()
    if attr_weights is not None:
        attr_weights = torch.from_numpy(attr_weights).float()
        if torch.cuda.is_available():
            attr_weights = attr_weights.cuda()


    attr_losses = []
    cat_losses = []
    for type in loss_types:

        if type == 'bce':
            #loss = nn.CrossEntropyLoss(weight=cat_weights)
            loss = CELoss(weight=cat_weights)
        elif type == 'dice':
            loss = DiceLoss(weight=cat_weights)
        elif type == 'logdice':
            loss = DiceLoss(weight=cat_weights, log=True)
        else:
            raise NotImplementedError("Loss type: {}".format(type))
        cat_losses.append(loss)

        attr_losses.append([])
        for i in range(num_attr):

            if attr_weights is not None:
                weight = attr_weights[i]
            else:
                weight = None

            if type == 'bce':
                loss = BCELoss(weight=weight)
            elif type == 'dice':
                loss = DiceLoss(weight=weight)
            elif type == 'logdice':
                loss = DiceLoss(weight=weight, log=True)
            else:
                raise NotImplementedError("Loss type: {}".format(type))

            attr_losses[-1].append(loss)

    def loss_func(input, target, alpha=None, rolled=None):
        total_loss = 0

        if num_cat > 0:
            for loss_coef, loss in zip(loss_coefs, cat_losses):
                total_loss += loss_coef * loss(input[:, :num_cat], target[:, 0],
                                               alpha=alpha, rolled=rolled)

        for loss_coef, loss_list in zip(loss_coefs, attr_losses):

            for i, loss in enumerate(loss_list):
                if num_cat > 0:
                    cur_target = target[:, 1 + i]
                else:
                    cur_target = target[:, i]
                if alpha is not None and rolled is not None:
                    cur_target = alpha * cur_target + (1-alpha) * cur_target[rolled, ...]
                total_loss += 1 / len(loss_list) * loss_coef * loss(input[:, num_cat + i], cur_target)

        return total_loss

    return loss_func
