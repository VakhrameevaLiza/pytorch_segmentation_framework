from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from core.get_loss_func import get_loss_func
from tensorboardX import SummaryWriter
import torch


lr_min = -8
lr_max = -2
num_iters = 500


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def get_der(x, y):
    n = x.shape[0]
    y_der = np.zeros(n-1)

    for i in range(n-1):
        y_der[i] = (y[i+1] - y[i]) / (np.log(x[i+1]) - np.log(x[i]))

    return y_der


def decoupled_weight_decay(model, lr, wd):
    for name, tensor in model.named_parameters():
        if 'bias' in name:
            continue
        tensor.data.add_(-wd * lr * tensor.data)


def find_lr(model, optimizer, train_dataset, config):
    model.cuda()
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              num_workers=config.num_workers, drop_last=True,
                              shuffle=True, pin_memory=True)

    loss_func = get_loss_func(config.loss_types, config.loss_coefs,
                              num_cat=train_dataset.num_cat, num_attr=train_dataset.num_attr,
                              cat_weights=config.cat_weights, attr_weights=config.attr_weights)

    dataloader = iter(train_loader)
    num_iters_per_epoch = len(dataloader)
    lr_range = np.logspace(lr_min, lr_max, num=num_iters)
    losses = np.zeros(num_iters)

    for i in tqdm(range(num_iters)):
        if i > 0 and i % num_iters_per_epoch == 0:
            dataloader = iter(train_loader)

        lr = lr_range[i]
        set_lr(optimizer, lr)

        batch = dataloader.next()

        device = torch.device("cuda")
        data = batch['data'].to(device, non_blocking=True)
        label = batch['label'].to(device, non_blocking=True)

        mixup_aug = config.mixup_aug
        mixup_coef = config.mixup_aug_beta
        if mixup_aug:
            bs = data.size(0)
            alpha = torch.from_numpy(np.random.beta(mixup_coef, mixup_coef, bs).astype(np.float32)).to(device)
            alpha = alpha[:, None, None, None]
            rolled = torch.from_numpy(np.roll(np.arange(bs), 1, axis=0))
            data_mix = (1. - alpha) * data + alpha * data[rolled, ...]
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

        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        decoupled_weight_decay(model, lr, config.weight_decay)

        loss = loss.item()
        losses[i] = loss

    sw_path = config.log_dir + '_find_lr'
    sw = SummaryWriter(sw_path)

    fig = plt.figure(figsize=(10,5))

    plt.plot(lr_range, losses)
    plt.grid()
    plt.xlabel("logscale lr", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.ylim((0., 2*losses[0]))
    plt.title("FindLr")
    plt.xscale("log")
    sw.add_figure(config.experiment_name, fig, 0)
    plt.close()
