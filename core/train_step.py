import torch
import numpy as np
from utils.iou import iou_func
from utils.losses.dice import dice_func
from utils.count_metric import count_metric
from core.helpers import set_lr, get_grad_norm, decoupled_weight_decay


def train_step(model, optimizer, dataloader, num_cat, num_attr,
               lr_schedule, weight_decay, global_step, loss_func,
               sw, device,
               mixup_aug, mixup_coef):

    lr = set_lr(optimizer, lr_schedule, global_step)

    batch = dataloader.next()
    data = batch['data'].to(device, non_blocking=True)
    label = batch['label'].to(device, non_blocking=True)

    if mixup_aug:
        bs = data.size(0)
        alpha = torch.from_numpy(np.random.beta(mixup_coef, mixup_coef, bs).astype(np.float32)).to(device)
        alpha = alpha[:, None, None, None]
        rolled = torch.from_numpy(np.roll(np.arange(bs), 1, axis=0))
        data_mix = alpha * data + (1 - alpha) * data[rolled, ...]
        output = model(data_mix)
    else:
        alpha = None
        rolled = None
        output = model(data)

    if isinstance(output, tuple):
        loss = 0
        for output_elem in output:
            loss += loss_func(output_elem, label, alpha=alpha, rolled=rolled)
        output = output[-1]
    else:
        loss = loss_func(output, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    decoupled_weight_decay(model, lr, weight_decay)

    loss = loss.item()

    cat_iou, attr_iou, attr_by_cl_iou = count_metric(output, label, iou_func, num_cat, num_attr)
    if num_cat <= 3:
        cat_dice, attr_dice, attr_by_cl_dice = count_metric(output, label, dice_func, num_cat, num_attr, pass_probs=True)
    else:
        cat_dice, attr_dice, attr_by_cl_dice = -1, -1, []

    # cat_iou = 0
    # attr_iou = 0
    # attr_by_cl_iou = []
    # cat_dice = 0
    # attr_dice = 0
    # attr_by_cl_dice = []

    d = {'fn': batch['fn'], "loss": loss,
         "cat_iou": cat_iou, "attr_iou": attr_iou, "attr_by_cl_iou": attr_by_cl_iou,
         "cat_dice": cat_dice, "attr_dice": attr_dice, "attr_by_cl_dice": attr_by_cl_dice}

    grad_norm = get_grad_norm(model)

    sw.add_scalar('loss', loss, global_step)
    sw.add_scalar('lr', lr, global_step)
    sw.add_scalar('grad_l2_norm', grad_norm, global_step)

    return d
