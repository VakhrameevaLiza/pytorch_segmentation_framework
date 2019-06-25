from easydict import EasyDict as edict
import numpy as np
import os

from models.Bisenet.Bisenet import BiSeNet
from models.SegRevnet_1 import SegRevnet
from models.Unet import Unet

experiments = []

######################################################
# 0
exp = edict()

exp.name = 'Ce_Bisenet_resnet34'

exp.net = BiSeNet
exp.net_kwargs = {'pretrained_model': "/home/vakhrameevaliza/.torch/models/resnet34-333f7ec4.pth",
                  'head': 'resnet34'}
exp.loss_types = ['bce']
exp.loss_coefs = [1.0]

exp.h = 512
exp.w = 512


exp.cat_weights = None
exp.attr_weights = None

exp.mixup_aug = False
exp.mixup_aug_beta = 0.4

exp.batch_size = 4
exp.num_epochs = 50

exp.lr_schedule_type = 'complex'
exp.lr_min = 10**-4.5
exp.lr_max = 10**-2.5
exp.lr = 10**-3


experiments.append(exp)
#####################################################

#####################################################
# 1
exp = edict()
exp.channels = 32
exp.num_layers_in_block = 4
exp.num_blocks = 10

exp.name = 'Ce_Revnet_{}_{}_{}'.format(exp.channels, exp.num_layers_in_block, exp.num_blocks)

exp.net = SegRevnet


exp.net_kwargs = {"channels": exp.channels,
                  "num_layers_in_block": exp.num_layers_in_block,
                  "num_blocks": exp.num_blocks}
exp.loss_types = ['bce']
exp.loss_coefs = [1.0]

exp.h = 512
exp.w = 512

exp.cat_weights = None
exp.attr_weights = None

exp.mixup_aug = True
exp.mixup_aug_beta = 0.4

exp.batch_size = 4
exp.num_epochs = 50

exp.lr_schedule_type = 'complex'
exp.lr_min = 10**-4
exp.lr_max = 10**-3
exp.lr = 10**-4

experiments.append(exp)
#####################################################

######################################################
# 2
exp = edict()

exp.name = 'Weighted_Ce_Bisenet_resnet34'

exp.net = BiSeNet
exp.net_kwargs = {'pretrained_model': "/home/vakhrameevaliza/.torch/models/resnet34-333f7ec4.pth",
                  'head': 'resnet34'}
exp.loss_types = ['bce']
exp.loss_coefs = [1.0]

exp.h = 512
exp.w = 512

exp.cat_weights = np.load('/home/vakhrameevaliza/segmentation_revnet/experiments/Fashion/cat_weights_2.npy')
exp.attr_weights=None

exp.mixup_aug = True
exp.mixup_aug_beta = 0.4

exp.batch_size = 4
exp.num_epochs = 50

exp.lr_schedule_type = 'complex'
exp.lr_min = 10**-4
exp.lr_max = 10**-3
exp.lr = 10**-4

experiments.append(exp)
#####################################################


######################################################
# 3
exp = edict()

exp.name = 'Ce_Bisenet_resnet34_mixup'

exp.net = BiSeNet
exp.net_kwargs = {'pretrained_model': "/home/vakhrameevaliza/.torch/models/resnet34-333f7ec4.pth",
                  'head': 'resnet34'}
exp.loss_types = ['bce']
exp.loss_coefs = [1.0]

exp.h = 512
exp.w = 512


exp.cat_weights = None
exp.attr_weights = None

exp.mixup_aug = True
exp.mixup_aug_beta = 0.4

exp.batch_size = 4
exp.num_epochs = 50

exp.lr_schedule_type = 'complex'
exp.lr_min = 10**-5
exp.lr_max = 10**-2.7
exp.lr = 10**-3


experiments.append(exp)
#####################################################

