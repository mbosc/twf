

import importlib
import os
import sys
import numpy as np
import socket
conf_path = os.getcwd()
sys.path.append(conf_path)
# sys.path.append(conf_path + '/datasets')
# sys.path.append(conf_path + '/backbone')
# sys.path.append(conf_path + '/models')
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_dataset
from models import get_model
from utils.training import train
import torch

import uuid
import datetime


def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    args = parser.parse_args()

    return args

import signal
import sys

sig_pause = False

def signal_handler(sig, frame):
    global sig_pause
    print('Signal received!')
    sig_pause = True



def main(args=None):
    import time
    start_time = time.time()

    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    
    # job number 
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    dataset = get_dataset(args)
    if args.scheduler == 'bic':
        dataset.N_CLASSES_PER_TASK = 20
        dataset.N_TASKS = 5
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    if args.model == 'joint':
        args.ignore_other_metrics=True
    model = get_model(args, backbone, loss, dataset.get_transform())
    
    import setproctitle
    setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))     

    train(model, dataset, args)

if __name__ == '__main__':
    main()
