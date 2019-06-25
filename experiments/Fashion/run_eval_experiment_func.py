import os
from datasets.FashionDataset import FashionDataset
from core.train_model import eval_model
import torch
from get_config import get_config

def load_model(config, net, mode='last'):
    file_name = os.path.join(config.training_states_dir, mode)
    states = torch.load(file_name, map_location='cuda:0')
    net.load_state_dict(states['model'])



def run_eval_experiment_func(exp):
    config = get_config(experiment_name=exp.name, lr_schedule_type=exp.lr_schedule_type,
                        loss_types=exp.loss_types, loss_coefs=exp.loss_coefs,
                        cat_weights=exp.cat_weights, attr_weights=exp.attr_weights,
                        batch_size=1, num_epochs=exp.num_epochs)

    valid_dataset = FashionDataset(config.dataset_path, 'validation', h=config.h, w=config.w, mean=config.mean,
                                   train_pct=0.8, valid_pct=0.2)

    net = exp.net(valid_dataset.num_classes, **exp.net_kwargs)
    load_model(config, net, "last")

    device = torch.device("cuda")
    net.to(device)

    results = eval_model(net, valid_dataset, config,
                         save_examples_flag=True, num_examples=50,
                         save_path=os.path.join(config.experiment_dir, 'examples'))

