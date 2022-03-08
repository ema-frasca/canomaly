from argparse import ArgumentParser
from datasets import get_all_datasets, get_dataset
from models import get_all_models, get_model
from utils.optims import add_optimizers_args, add_optim_args
from utils.writer import writer


def parse_args():
    parser = ArgumentParser(description='canomaly', allow_abbrev=False)

    # torch.set_num_threads(4)
    add_management_args(parser)
    add_experiment_args(parser)
    args, unknown_args = parser.parse_known_args()

    model = get_model(args.model)
    dataset = get_dataset(args.dataset)
    model.add_model_args(parser)
    dataset.add_dataset_args(parser)
    add_optim_args(args.optim, parser)
    add_approach_args(parser, args.approach)

    args = parser.parse_args()

    return args


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=get_all_datasets(),
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    add_optimizers_args(parser)

    writer.dir_args.append('dataset')

    # these are model / dataset args
    parser.add_argument('--approach', type=str, choices=['continual', 'joint', 'splits'], default='continual',
                        help='Type of training.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, required=True,
                        help='The number of epochs for each task.')


def add_approach_args(parser: ArgumentParser, approach: str) -> None:
    if approach == 'joint':
        parser.add_argument('--per_task', action='store_true',
                            help='Compute joint training for every task.')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')

    parser.add_argument('--logs', action='store_true',
                        help='Enable full logging.')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging.')
    # parser.add_argument('--validation', action='store_true',
    #                     help='Test on the validation set')
