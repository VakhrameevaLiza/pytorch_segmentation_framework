from get_config import get_config
get_config('')

from experiments import experiments
from run_experiment_func import run_experiment_func
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('experiment_id', metavar='N', type=int)
parser.add_argument('--device_id', metavar='d', type=int)

args = parser.parse_args()
torch.cuda.set_device(args.device_id)

experiment_id = args.experiment_id
experiment = experiments[experiment_id]
print(experiment.name)
run_experiment_func(experiment)