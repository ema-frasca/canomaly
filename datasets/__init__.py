import os
import importlib
from typing import Type
from datasets.utils.canomaly_dataset import CanomalyDataset


def get_all_datasets():
    return [dataset.split('.')[0] for dataset in os.listdir('datasets')
            if '__' not in dataset and dataset.endswith('.py')]


def get_dataset(dataset: str) -> Type[CanomalyDataset]:
    mod = importlib.import_module('datasets.' + dataset)
    class_name = {x.lower(): x for x in mod.__dir__()}[dataset.replace('_', '')]
    return getattr(mod, class_name)
