import os
from get_config import get_config
get_config('')
import torch
from torch.optim import Adam
from datasets.FashionDataset import FashionDataset
from utils.find_lr import find_lr


def run_findlr_func(exp):

    config = get_config(experiment_name=exp.name, lr_schedule_type=exp.lr_schedule_type, lr=exp.lr,
                        loss_types=exp.loss_types, loss_coefs=exp.loss_coefs,
                        cat_weights=exp.cat_weights, attr_weights=exp.attr_weights,
                        batch_size=exp.batch_size, num_epochs=exp.num_epochs,
                        mixup_aug=exp.mixup_aug, mixup_aug_beta=exp.mixup_aug_beta
                        )

    train_dataset = FashionDataset(config.dataset_path, 'training', h=config.h, w=config.w, mean=config.mean,
                                   train_pct=1., valid_pct=0.15)
    net = exp.net(train_dataset.num_classes, **exp.net_kwargs)
    optimizer = Adam(net.parameters(), weight_decay=0)
    device = torch.device("cuda:0")
    find_lr(net, optimizer, train_dataset, config)

    del net
    del optimizer
    del device

    torch.cuda.empty_cache()
