from argparse import ArgumentParser, Namespace
from torch.optim import SGD, Adam


def add_optimizers_args(parser: ArgumentParser) -> None:
    parser.add_argument('--optim', type=str, default='SGD',
                        choices=['SGD', 'Adam'],
                        help='Optimizer used on the network.')

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')


def add_optim_args(optim: str, parser: ArgumentParser) -> None:
    if optim == 'SGD':
        return
    if optim == 'Adam':
        return


def get_optim(args: Namespace):
    if args.optim == 'SGD':
        return SGD, {'lr': args.lr}
    if args.optim == 'Adam':
        return Adam, {'lr': args.lr}