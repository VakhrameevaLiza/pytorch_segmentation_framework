import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.ensure_dir import ensure_dir_exists, ensure_dir_exists_and_empty
from tensorboardX import SummaryWriter
from core.get_loss_func import get_loss_func
from core.train_step import train_step
from core.eval_model import eval_model
import numpy as np
import time
import os
from collections import defaultdict
from core.helpers import get_lr_schedule, process_metrics, get_mean_iou, write_logs,create_state_dict
from utils.save_examples import save_examples



def train_model(model, optimizer, train_dataset, valid_dataset, config, continue_from_last=False):


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              num_workers=config.num_workers, drop_last=True,
                              shuffle=True, pin_memory=True)

    niters_per_epoch = len(train_dataset) // config.batch_size
    num_epochs = config.num_epochs

    lr_schedule = get_lr_schedule(config, niters_per_epoch, num_epochs)

    sw_path = config.log_dir
    training_states_dir = config.training_states_dir
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda")
    else:
        raise NotImplementedError('Training on cpu')

    if continue_from_last:
        state_dict = torch.load(os.path.join(training_states_dir, "last"))
        global_step = state_dict['global_step']
        continue_from_epoch = state_dict['epoch'] + 1
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optim'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    else:
        continue_from_epoch = 0
        global_step = 0
        ensure_dir_exists_and_empty(sw_path)
        ensure_dir_exists_and_empty(training_states_dir)

    sw = SummaryWriter(sw_path)
    model.to(device)

    num_cat = train_dataset.num_cat
    num_attr = train_dataset.num_attr

    loss_func = get_loss_func(config.loss_types, config.loss_coefs,
                              num_cat=num_cat, num_attr=num_attr,
                              cat_weights=config.cat_weights, attr_weights=config.attr_weights)

    best_iou = 0
    num_epochs_wait = 0
    start_train_time = time.time()

    for epoch in range(continue_from_epoch, num_epochs):
        last_saving_time = time.time()
        model.train()
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        with tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format) as pbar:
            dataloader = iter(train_loader)

            train_metrics = defaultdict(list)

            for idx in pbar:
                start = time.time()

                batch_metrics = train_step(model, optimizer, dataloader, num_cat, num_attr,
                                           lr_schedule, config.weight_decay,
                                           global_step, loss_func, sw, device, config.mixup_aug, config.mixup_aug_beta)

                mean_batch_iuo = get_mean_iou(num_cat, num_attr, batch_metrics)

                delta_time = time.time() - start

                sw.add_scalar('sec_to_batch', delta_time, global_step)
                sw.add_scalar('train_batch_iou', mean_batch_iuo, global_step)
                sw.add_scalar('train_batch_cat_iou', batch_metrics['cat_iou'], global_step)
                sw.add_scalar('train_batch_attr_iou', batch_metrics['attr_iou'], global_step)

                global_step += 1
                pbar.set_description(str(idx), refresh=False)
                pbar_desc = "Train: Epoch {:03d}/{:03d}; Iter {:04d}/{:04d}; iou = {:05.2f}, loss = {:.2f}; ".format(epoch+1,
                                num_epochs, idx+1, niters_per_epoch, mean_batch_iuo, batch_metrics["loss"])
                pbar.set_description(pbar_desc, refresh=False)

                for k, v in batch_metrics.items():
                    train_metrics[k].append(v)
                train_metrics['iou'].append(mean_batch_iuo)
                if idx % 1000 == 0:
                    last_saving_time = time.time()
                    state_dict = create_state_dict(model, optimizer, epoch, global_step, None, config)
                    torch.save(state_dict, os.path.join(training_states_dir, "last"))
                    #print("\n Successfully save")

        process_metrics(train_metrics)
        write_logs(sw, "train", train_metrics, epoch)

        val_metrics = eval_model(model, valid_dataset, config,
                                 save_examples_flag=False, num_examples=20,
                                 save_path=os.path.join(config.experiment_dir, 'examples'))

        iou = get_mean_iou(num_cat, num_attr, val_metrics)

        if config.lr_schedule_type == 'complex':
            lr_schedule.update_val_stats(iou)

        state_dict = create_state_dict(model, optimizer, epoch, global_step, val_metrics, config)
        torch.save(state_dict, os.path.join(training_states_dir, "last"))

        if iou > best_iou:
            best_iou = iou
            torch.save(state_dict, os.path.join(config.training_states_dir, "best"))
            num_epochs_wait = 0
        else:
            num_epochs_wait += 1

        write_logs(sw, "valid", val_metrics, epoch)

        if num_epochs_wait > config.max_epochs_wait or epoch == config.num_epochs-1:
            results = {}

            for k, v in train_metrics.items():
                results['train_' + k] = v
            for k, v in val_metrics.items():
                results['valid_' + k] = v
            results['num_epochs'] = epoch + 1
            results['train_time_min'] = (time.time() - start_train_time) / 60

            return results
