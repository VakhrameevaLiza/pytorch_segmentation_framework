import torch
from utils.iou import iou_func
import numpy as np
from utils.losses.dice import dice_func
from utils.count_metric import count_metric


def eval_step(model, dataloader, loss_func, num_cat, num_attr):
    cuda = torch.cuda.is_available()

    batch = dataloader.next()
    data = batch['data']
    label = batch['label']

    if cuda:
        data = batch['data'].cuda(non_blocking=True)
        label = batch['label'].cuda(non_blocking=True)

    output = model(data)
    if isinstance(output, tuple):
        loss = 0
        for output_elem in output:
            loss += loss_func(output_elem, label).item()
        output = output[-1]
    else:
        loss = loss_func(output, label).item()
    cat_iou, attr_iou, attr_by_cl_iou = count_metric(output, label, iou_func, num_cat, num_attr)

    if num_cat <= 3:
        cat_dice, attr_dice, attr_by_cl_dice = count_metric(output, label, dice_func, num_cat, num_attr, pass_probs=True)
    else:
        cat_dice, attr_dice, attr_by_cl_dice = -1, -1, []

    d = {'fn': batch['fn'], "loss": loss,
         "cat_iou": cat_iou, "attr_iou": attr_iou, "attr_by_cl_iou": attr_by_cl_iou,
         "cat_dice": cat_dice, "attr_dice": attr_dice, "attr_by_cl_dice": attr_by_cl_dice}

    return d
