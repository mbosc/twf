

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from datetime import datetime

def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, required=True,
                        help='The number of epochs for each task.')
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=[None, 'simple'])
    parser.add_argument('--scheduler_rate', type=float, default=None)
    parser.add_argument('--opt_steps', type=int, nargs='+', default=None)
    

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', action='store_true')

    parser.add_argument('--ignore_other_metrics', action='store_true',
                        help='disable additional metrics')

    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--start_from', type=int, default=None, help="Task to start from")
    parser.add_argument('--stop_after', type=int, default=None, help="Task limit")


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, required=True,
                        help='The batch size of the memory buffer.')

def add_aux_dataset_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used to load initial (pretrain) checkpoint
    :param parser: the parser instance
    """
    parser.add_argument('--pre_epochs', type=int, default=200,
                        help='pretrain_epochs.')
    parser.add_argument('--pre_dataset', type=str, required=True,
                        choices=['cifar100', 'tinyimgR', 'imagenet'])
    parser.add_argument('--load_cp', type=str, default=f'/tmp/checkpoint_{datetime.now().timestamp()}.pth')
    parser.add_argument('--stop_after_prep', action='store_true')
