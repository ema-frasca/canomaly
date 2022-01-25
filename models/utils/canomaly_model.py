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
from utils.logger import logger


class CanomalyModel:
    NAME = None

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        pass

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        self.args = args
        self.dataset = dataset
        self.device = config.device
        self.config = config

        Optim, optim_args = get_optim(args)
        self.Optimizer = Optim
        self.optim_args = optim_args
        self.full_log = vars(args)
        self.full_log['results'] = {}
        self.full_log['knowledge'] = {}

    @abstractmethod
    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int) -> float:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def test_step(self, test_loader: DataLoader, task: int):
        self.full_log['results'][str(task)] = {'targets': [], 'rec_errs': []}
        progress = logger.get_tqdm(test_loader, f'TEST on task {task}')
        for X, y in progress:
            self.full_log['results'][str(task)]['targets'].extend(y.tolist())
            X.to(self.device)
            rec_errs = self.forward(X)
            self.full_log['results'][str(task)]['rec_errs'].extend(rec_errs.tolist())
        progress.close()

    def train_on_task(self, task_loader: DataLoader, task: int):
        progress = logger.get_tqdm(task_loader, f'TRAIN on task {task}')
        for x, y in progress:
            loss = self.train_on_batch(x, y, task)
            progress.set_postfix({'loss': loss})
        progress.close()

    def train_on_dataset(self):
        logger.log(vars(self.args))
        if self.args.joint:
            self.train_on_task(self.dataset.joint_loader(), 0)
            self.test_step(self.dataset.test_loader(), 0)
            self.full_log['knowledge']['0'] = self.dataset.last_seen_classes.copy()
        else:
            for i, task_dl in enumerate(self.dataset.task_loader()):
                self.train_on_task(task_dl, i)
                self.full_log['knowledge'][str(i)] = self.dataset.last_seen_classes.copy()
                # evaluate on test
                self.test_step(self.dataset.test_loader(), i)

    def print_log(self):
        if self.args.logs:
            writer.write_log(self.full_log)
