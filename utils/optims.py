from argparse import ArgumentParser, Namespace
from torch.optim import SGD, Adam


def add_optimizers_args(parser: ArgumentParser) -> None:
    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help='Optimizer used on the network.')

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')


def add_optim_args(optim: str, parser: ArgumentParser) -> None:
    if optim == 'sgd':
        return
    if optim == 'adam':
        return


def get_optim(args: Namespace):
    if args.optim == 'sgd':
        return SGD, {'lr': args.lr}
    if args.optim == 'adam':
        return Adam, {'lr': args.lr}