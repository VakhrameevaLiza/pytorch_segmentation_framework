import os

from get_config import get_config
get_config('')

from utils.log_experiment_results import log_results

from torch.optim import Adam
from datasets.FashionDataset import FashionDataset
from core.train_model import train_model


def run_experiment_func(exp):

    config = get_config(experiment_name=exp.name, lr_schedule_type=exp.lr_schedule_type, lr=exp.lr,
                        loss_types=exp.loss_types, loss_coefs=exp.loss_coefs,
                        cat_weights=exp.cat_weights, attr_weights=exp.attr_weights,
                        batch_size=exp.batch_size, num_epochs=exp.num_epochs,
                        mixup_aug=exp.mixup_aug, mixup_aug_beta=exp.mixup_aug_beta,
                        h=exp.h, w=exp.w
                        )

    train_dataset = FashionDataset(config.dataset_path, 'training', h=config.h, w=config.w, mean=config.mean,
                                   train_pct=0.8, valid_pct=0.2)
    valid_dataset = FashionDataset(config.dataset_path, 'validation', h=config.h, w=config.w, mean=config.mean,
                                   train_pct=0.8, valid_pct=0.2)

    net = exp.net(train_dataset.num_classes, **exp.net_kwargs)
    optimizer = Adam(net.parameters(), weight_decay=0)

    results = train_model(net, optimizer, train_dataset, valid_dataset, config)
    results['name'] = exp.name
    results['model'] = "{}({}x{})".format(net.name, config.h, config.w) #'Unet(512x512)'
    results['loss'] = '_'.join(str(coef) + 'x' + loss for coef, loss in zip(exp.loss_coefs, exp.loss_types))
    results['lr_schedule'] = exp.lr_schedule_type
    results['weights'] = 'True' if exp.weights is not None else 'False'
    log_results(results, 'log.csv')


