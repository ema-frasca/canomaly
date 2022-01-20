import os
import importlib


def get_all_datasets():
    return [dataset.split('.')[0] for dataset in os.listdir('datasets')
            if '__' not in dataset and dataset.endswith('.py')]


def get_dataset(args):
    mod = importlib.import_module('models.' + args.dataset)
    class_name = {x.lower(): x for x in mod.__dir__()}[args.dataset.replace('_', '')]
    dataset_class = getattr(mod, class_name)
    return dataset_class(args)
