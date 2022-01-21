import torch.nn as nn
from torch.optim import SGD
import torch
from torch.utils.data import DataLoader
import torchvision
from abc import abstractmethod
from argparse import Namespace, ArgumentParser
from utils.config import config
from datasets.utils.canomaly_dataset import CanomalyDataset


class CanomalyModel():
    NAME = None

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        pass

    def __init__(self, args: Namespace):
        self.args = args
        self.device = config.device

    @abstractmethod
    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        pass

    def train_on_task(self, task_loader: DataLoader, task: int):
        for x, y in task_loader:
            self.train_on_batch(x, y, task)

    def train_on_dataset(self, dataset: CanomalyDataset):
        for i, task_dl in enumerate(dataset.task_loader()):
            self.train_on_task(task_dl, i)
            # evaluate on test
