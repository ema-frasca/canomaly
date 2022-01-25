import torch.nn as nn
from torch.optim import SGD
import torch
from torch.utils.data import DataLoader
import torchvision
from abc import abstractmethod
from argparse import Namespace, ArgumentParser
from utils.config import config
from datasets.utils.canomaly_dataset import CanomalyDataset
from utils.optims import get_optim
from utils.writer import writer


class CanomalyModel():
    NAME = None

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        pass

    def __init__(self, args: Namespace):
        self.args = args
        self.device = config.device
        self.config = config

        Optim, optim_args = get_optim(args)
        self.Optimizer = Optim
        self.optim_args = optim_args
        self.full_log = vars(args)
        self.full_log['results'] = {}
        self.full_log['knowledge'] = {}

    @abstractmethod
    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def test_step(self, test_loader: DataLoader, task: int):
        self.full_log['results'][str(task)] = {'targets': [], 'rec_errs': []}
        for X, y in test_loader:
            self.full_log['results'][str(task)]['targets'].extend(y.tolist())
            X.to(self.device)
            rec_errs = self.forward(X)
            self.full_log['results'][str(task)]['rec_errs'].extend(rec_errs.tolist())

    def train_on_task(self, task_loader: DataLoader, task: int):
        for x, y in task_loader:
            self.train_on_batch(x, y, task)

    def train_on_dataset(self, dataset: CanomalyDataset):
        if self.args.joint:
            self.train_on_task(dataset.joint_loader(), 0)
            self.test_step(dataset.test_loader(), 0)
            self.full_log['knowledge']['0'] = dataset.last_seen_classes.copy()
        else:
            for i, task_dl in enumerate(dataset.task_loader()):
                self.train_on_task(task_dl, i)
                self.full_log['knowledge'][str(i)] = dataset.last_seen_classes.copy()
                # evaluate on test
                self.test_step(dataset.test_loader(), i)

    def print_log(self):
        if self.args.logs:
            writer.write_log(self.full_log)
