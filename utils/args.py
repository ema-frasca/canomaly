import importlib
from argparse import ArgumentParser
from datasets import get_all_datasets, get_dataset
from models import get_all_models, get_model
from utils.config import config


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

    args = parser.parse_args()

    if args.seed is not None:
        config.set_seed(args.seed)

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

    # these are model / dataset args
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, required=True,
                        help='The number of epochs for each task.')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')

    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging.')
    # parser.add_argument('--validation', action='store_true',
    #                     help='Test on the validation set')
