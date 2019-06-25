from easydict import EasyDict as edict
import sys
import os

def get_config(experiment_name, lr_schedule_type='complex',
               loss_types=['bce'], loss_coefs=[1.0],
               h=512, w=512,
               lr=10e-5, lr_min=10e-5, lr_max=10e-4, batch_size=4,
               mixup_aug=False, mixup_aug_beta=1.,
               #weights=None,
               cat_weights=None, attr_weights=None,
               channels=32, num_layers_in_block=4, num_blocks=5,
               num_epochs=5):


    config = edict()
    config.seed = 42

    config.repo_name = 'segmentation_revnet'
    config.experiment_name = experiment_name

    abs_dir = os.path.realpath('.')
    config.root_dir = abs_dir[:abs_dir.index(config.repo_name) + len(config.repo_name)]

    if 'vakhrameevaliza' in config.root_dir:
        config.dataset_path = '/home/vakhrameevaliza/fashion2019'
    else:
        config.dataset_path = '/home/liza/fashion2019'


    config.experiment_dir = os.path.join(config.root_dir, 'experiments', 'Fashion', config.experiment_name)

    config.log_dir = os.path.join(config.experiment_dir, 'logs')
    config.training_states_dir = os.path.join(config.experiment_dir, 'training_states')
    config.predictions_dir = os.path.join(config.experiment_dir, 'predictions')

    def add_path(path):
        if path not in sys.path:
            sys.path.insert(0, path)
    add_path(os.path.join(config.root_dir))
    add_path(os.path.join(config.root_dir, 'datasets'))
    add_path(os.path.join(config.root_dir, 'models'))
    add_path(os.path.join(config.root_dir, 'core'))
    add_path(os.path.join(config.root_dir, 'utils'))
    add_path(os.path.join(config.root_dir, 'experiments'))

    # data
    config.h = h
    config.w = w
    config.c = 3
    config.mean = 0.5

    config.batch_size = batch_size
    config.num_workers = 8
    config.num_epochs = num_epochs

    config.lr_schedule_type = lr_schedule_type
    config.lr = lr
    config.lr_min = lr_min
    config.lr_max = lr_max
    config.cycle_len_in_epochs = 5
    config.power = 0.9
    config.weight_decay = 1e-3
    config.loss_types = loss_types
    config.loss_coefs = loss_coefs
    config.max_epochs_wait = 10

    config.mixup_aug = mixup_aug
    config.mixup_aug_beta = mixup_aug_beta

    config.cat_weights = cat_weights
    config.attr_weights = attr_weights

    # for segmentation revnet
    config.channels = channels
    config.num_layers_in_block = num_layers_in_block
    config.num_blocks = num_blocks

    return config
