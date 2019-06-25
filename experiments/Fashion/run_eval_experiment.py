from get_config import get_config
get_config('')

from experiments import experiments
from run_eval_experiment_func import run_eval_experiment_func

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('experiment_id', metavar='N', type=int)
parser.add_argument('--device_id', metavar='d', type=int)
args = parser.parse_args()

device_id = args.device_id
torch.cuda.set_device(device_id)

experiment_id = args.experiment_id
run_eval_experiment_func(experiments[experiment_id])