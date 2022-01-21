from abc import abstractmethod
from argparse import Namespace, ArgumentParser
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Union
from torchvision import datasets
import numpy as np
import socket
import torch
import os
from utils.config import config
from typing import Generator


class CanomalyDataset:
    NAME: str = None
    N_CLASSES: int = None

    @staticmethod
    def add_dataset_args(parser: ArgumentParser):
        pass

    def __init__(self, args: Namespace):
        self.config = config
        self.args = args

        self.classes_per_task = None
        self.n_tasks = None

        self.train_dataset = None
        self.test_dataset = None

    @abstractmethod
    def _get_task_dataset(self) -> Generator[Dataset]:
        pass

    def task_loader(self):
        for ds in self._get_task_dataset():
            yield DataLoader(ds, batch_size=self.args.batch_size, shuffle=True)

    def joint_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False)
