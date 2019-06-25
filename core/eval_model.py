import torch
#from utils.count_iou import count_iou
from tqdm import tqdm
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.ensure_dir import ensure_dir_exists_and_empty
from utils.save_examples import save_examples
import numpy as np
from core.eval_step import eval_step
from core.get_loss_func import get_loss_func
from collections import defaultdict
from core.helpers import get_mean_iou, process_metrics
from torch.utils.data import DataLoader


def eval_model(model, valid_dataset, config,
               save_examples_flag=True, num_examples=5,
               save_path=None
               ):
    valid_loader = DataLoader(valid_dataset, batch_size=1,
                              num_workers=config.num_workers,
                              shuffle=False, pin_memory=True)

    model.eval()
    if save_examples_flag:
        assert save_path

    niters = len(valid_dataset)
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

    num_cat = valid_dataset.num_cat
    color_cat_func = valid_dataset.cat_color

    num_attr = valid_dataset.num_attr
    color_attr_func = valid_dataset.attr_color

    loss_func = get_loss_func(config.loss_types, config.loss_coefs,
                              num_cat=num_cat, num_attr=num_attr,
                              cat_weights=config.cat_weights, attr_weights=config.attr_weights)

    valid_metrics = defaultdict(list)
    with tqdm(range(niters), file=sys.stdout, bar_format=bar_format) as pbar:
        dataloader = iter(valid_loader)
        for idx in pbar:
            batch_metrics = eval_step(model, dataloader, loss_func, num_cat, num_attr)
            for k, v in batch_metrics.items():
                valid_metrics[k].append(v)
            mean_batch_iuo = get_mean_iou(num_cat, num_attr, batch_metrics)
            valid_metrics['iou'].append(mean_batch_iuo)

            pbar_desc = "Valid: {}; Iter {:04d}/{:04d}; iou = {:05.2f}, loss = {:.2f}; ".format(13*' ', idx + 1, niters,
                                                                                                mean_batch_iuo,
                                                                                                batch_metrics["loss"])
            pbar.set_description(pbar_desc, refresh=False)

    if save_examples_flag:
        names_scores = zip(range(niters), valid_metrics['iou'])
        names_scores = sorted(names_scores, key=lambda t: -t[1])

        save_path_best = os.path.join(save_path, 'best')
        ensure_dir_exists_and_empty(save_path_best)
        save_examples(num_cat, num_attr,
                      names_scores[:num_examples], model, valid_dataset.get_by_ind,
                      color_cat_func, color_attr_func, save_path_best, config)

        save_path_worst = os.path.join(save_path, 'worst')
        ensure_dir_exists_and_empty(save_path_worst)
        save_examples(num_cat, num_attr,
                      names_scores[-num_examples:], model, valid_dataset.get_by_ind,
                      color_cat_func, color_attr_func, save_path_worst, config)

    process_metrics(valid_metrics)
    return valid_metrics


