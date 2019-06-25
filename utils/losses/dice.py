from torch import nn
from torch.nn import functional as F
import torch
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=-100, weight=None, log=False):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.log = log

    def forward(self, input, target):
        if len(input.shape) == 4:
            probs = F.softmax(input, dim=1)
        else:
            input = torch.sigmoid(input)[:, None, :, :]
            probs = torch.cat([1. - input, input], dim=1)

        tmp = torch.zeros_like(probs)
        if torch.cuda.is_available():
            tmp = tmp.cuda()

        one_hot_target = tmp.scatter_(1, target[:, None, :, :], 1.)

        intersection = torch.sum(probs * one_hot_target, dim=(2,3))
        union = torch.sum(probs * one_hot_target + one_hot_target, dim=(2,3))
        
        if self.weight is not None:
            intersection *= self.weight
            union *= self.weight

        intersection = torch.sum(intersection, dim=1)
        union = torch.sum(union, dim=1)

        smooth = 1.
        dice_loss = torch.mean(1 - 2 * (intersection + smooth) / (union + smooth))

        if self.log:
            dice_loss = torch.log(dice_loss)

        return dice_loss.item()


def dice_func(predict, target, num_classes, ignore_bg=False):
    return [DiceLoss()(predict, target)]

# def count_dice(input, target, num_cat=None, num_attr=None, num_cat_with_bg=False):
#     dice_func = DiceLoss()
#
#     if len(target.shape) == 3:
#         num_classes = input.shape[1]
#         return dice_func(input.argmax(dim=1), target, num_classes)
#
#     elif num_cat is not None:
#
#             if not num_cat_with_bg:
#                 num_cat += 1
#             cat_iou = dice_func(input[:num_cat].argmax(dim=1), target[:,0], num_cat)
#             ious = []
#             for i in range(num_attr):
#                 cl_probs = torch.sigmoid(input[:, num_cat+i:num_cat+(i+1)])
#                 bg_probs = 1. - cl_probs
#                 probs = torch.cat([bg_probs, cl_probs], dim=1)
#                 iou = dice_func(probs.argmax(dim=1), target[:, num_cat+i], 2)[0]
#                 ious.append(iou)
#             attr_iou = np.mean(ious)
#             return cat_iou, attr_iou
#
#     else:
#         num_classes = target.shape[1]
#         dice_coefs = []
#         for i in range(num_classes):
#             dice = dice_func(input[:, i], target[:, i]).item()
#             dice_coefs.append(dice)
#         return np.mean(dice_coefs), dice_coefs



if __name__ == '__main__':
    dice_loss = DiceLoss(4)

    input = torch.from_numpy( np.array([[ [1, 1], [1,1] ]])).float()
    target = torch.from_numpy( np.array([[ [0, 1], [1, 0] ]]))

    loss = dice_loss(input, target)

