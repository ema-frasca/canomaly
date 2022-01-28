import torch
from torch.utils.data import DataLoader
from abc import abstractmethod
from argparse import Namespace, ArgumentParser
from utils.config import config
from datasets.utils.canomaly_dataset import CanomalyDataset
from utils.optims import get_optim
from utils.writer import writer
from utils.logger import logger
from utils.metrics import reconstruction_error, compute_exp_metrics, reconstruction_confusion_matrix, compute_task_auc
from random import random


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
        self.full_log = {**vars(args), 'results': {}, 'knowledge': {}}

        self.net: torch.nn.Module = None

    @abstractmethod
    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int) -> float:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def net_train(self):
        self.net.train()

    def net_eval(self):
        self.net.eval()

    def test_step(self, test_loader: DataLoader, task: int):
        self.net_eval()
        self.full_log['results'][str(task)] = {'targets': [], 'rec_errs': [], 'images': []}
        progress = logger.get_tqdm(test_loader, f'TEST on task {task+1}')
        images_sample = {}
        for X, y in progress:
            self.full_log['results'][str(task)]['targets'].extend(y.tolist())
            X = X.to(self.device)
            outs = self.forward(X)
            rec_errs = reconstruction_error(X, outs)
            self.full_log['results'][str(task)]['rec_errs'].extend(rec_errs.tolist())
            if len(images_sample) < self.dataset.N_CLASSES and random() < 0.1:
                for i in range(len(y)):
                    if str(y[i].item()) not in images_sample:
                        images_sample[str(y[i].item())] = {'original': X[i].tolist(),
                                                           'reconstruction': outs[i].tolist()}
        images_sample = dict(sorted(images_sample.items()))
        for label in images_sample:
            self.full_log['results'][str(task)]['images'].append({'label': label, **images_sample[label]})

        auc = compute_task_auc(
            self.full_log['results'][str(task)]['targets'],
            self.full_log['results'][str(task)]['rec_errs'],
            [label for knowledge in self.full_log['knowledge'].values() for label in knowledge]
        )
        logger.log(f'TEST on task {task+1} - roc_auc = {auc}')

    def train_on_task(self, task_loader: DataLoader, task: int):
        self.net_train()
        for e in range(self.args.n_epochs):
            keep_progress = True  # if e == self.args.n_epochs - 1 else False
            progress = logger.get_tqdm(task_loader,
                                       f'TRAIN on task {task+1}/{self.dataset.n_tasks} - epoch {e+1}/{self.args.n_epochs}',
                                       leave=keep_progress)
            for x, y in progress:
                loss = self.train_on_batch(x.to(self.device), y.to(self.device), task)
                progress.set_postfix({'loss': loss})

    def train_on_dataset(self):
        logger.log(vars(self.args))
        if self.args.joint:
            self.train_on_task(self.dataset.joint_loader(), 0)
            self.full_log['knowledge']['0'] = self.dataset.last_seen_classes.copy()
            self.test_step(self.dataset.test_loader(), 0)
        else:
            for i, task_dl in enumerate(self.dataset.task_loader()):
                self.train_on_task(task_dl, i)
                self.full_log['knowledge'][str(i)] = self.dataset.last_seen_classes.copy()
                # evaluate on test
                self.test_step(self.dataset.test_loader(), i)

    def print_results(self):
        if self.args.logs:
            writer.write_log(self.full_log)

        res_log = {**vars(self.args)}
        auc_f, auc_a, auc_pt = compute_exp_metrics(self.full_log)
        cmatrix = reconstruction_confusion_matrix(self.full_log)
        res_log['auc_final'] = auc_f
        res_log['auc_average'] = auc_a
        res_log['auc_per_task'] = auc_pt
        res_log['conf_matrix_per_task'] = cmatrix

        writer.write_log(res_log, result=True)
