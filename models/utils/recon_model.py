from typing import Union
import torch
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader
from abc import abstractmethod
from argparse import Namespace, ArgumentParser

from models.utils.canomaly_model import CanomalyModel
from utils.config import config
from datasets.utils.canomaly_dataset import CanomalyDataset
from utils.optims import get_optim
from utils.writer import writer
from utils.logger import logger
from utils.metrics import (reconstruction_error, compute_exp_metrics, reconstruction_confusion_matrix, compute_task_auc,
                           print_reconstructed_vs_true)
from random import random


def mean(x):
    return sum(x) / len(x)


class ReconModel(CanomalyModel):

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super(ReconModel, self).__init__(args, dataset)

    def anomaly_score(self, recs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        return F.mse_loss(recs, x, reduction='none').mean(dim=[i for i in range(len(recs.shape))][1:])

    def test_step(self, test_loader: DataLoader, task: int):
        self.net_eval()
        self.full_log['results'][str(task)] = {'targets': [], 'rec_errs': [], 'images': []}
        progress = logger.get_tqdm(test_loader, f'TEST on task {task + 1}')
        images_sample = {}
        for X, y in progress:
            self.full_log['results'][str(task)]['targets'].extend(y.tolist())
            X = X.to(self.device)
            outs = self.forward(X)
            rec_errs = self.anomaly_score(outs, X)
            self.full_log['results'][str(task)]['rec_errs'].extend(rec_errs.tolist())
            if len(images_sample) < self.dataset.N_CLASSES and random() < 0.1:
                for i in range(len(y)):
                    if str(y[i].item()) not in images_sample:
                        images_sample[str(y[i].item())] = {
                            'original': X[i].tolist(),
                            'reconstruction': outs[i].tolist() if isinstance(outs, torch.Tensor)
                            else outs[0][i].tolist()}
                        # print_reconstructed_vs_true(outs[i], X[i], y[i], (28, 28))
        images_sample = dict(sorted(images_sample.items()))
        for label in images_sample:
            self.full_log['results'][str(task)]['images'].append({'label': label, **images_sample[label]})

        rec_mean = mean(self.full_log['results'][str(task)]['rec_errs'])
        self.full_log['results'][str(task)]['rec_mean'] = rec_mean
        logger.log(f'TEST on task {task + 1} - reconstruction error mean = {rec_mean}')

    def train_on_task(self, task_loader: DataLoader, task: int):
        self.cur_task = task
        self.net_train()
        for e in range(self.args.n_epochs):
            keep_progress = True  # if e == self.args.n_epochs - 1 else False
            progress = logger.get_tqdm(task_loader,
                                       f'TRAIN on task {task + 1}/{self.dataset.n_tasks} - epoch {e + 1}/{self.args.n_epochs}',
                                       leave=keep_progress)
            for x, y in progress:
                loss = self.train_on_batch(x.to(self.device), y.to(self.device), task)
                progress.set_postfix({'loss': loss})
            self.validate()
            self.scheduler_step()

    def train_on_dataset(self):
        logger.log(vars(self.args))
        # logger.log(self.net)
        loader = self.dataset.joint_loader if self.joint else self.dataset.task_loader
        freezed_params = [param.data.clone() for param in self.net.parameters()] if self.joint else []
        for i, task_dl in enumerate(loader()):
            if self.joint:
                for pidx, param in enumerate(self.net.parameters()):
                    param.data = freezed_params[pidx].clone()

            self.train_on_task(task_dl, i)
            self.full_log['knowledge'][str(i)] = self.dataset.last_seen_classes.copy()
            # evaluate on test
            self.test_step(self.dataset.test_loader(), i)

    def print_results(self):
        if self.args.logs:
            writer.write_log(self.full_log)

        res_log = {**vars(self.args)}
        cmatrix = reconstruction_confusion_matrix(self.full_log)
        res_log['conf_matrix_per_task'] = cmatrix
        writer.write_log(res_log, result=True)



from utils.metrics import print_reconstructed_vs_true
import numpy as np
# #
# n = 0
# nu, log = torch.tensor([0.4914, 0.4822, 0.4465]), torch.tensor([0.2470, 0.2435, 0.2615])
# # grayscale:
# # nu, log = torch.tensor([(0.4914+0.4822+0.4465)/3]), torch.tensor([(0.2470+0.2435+0.2615)/3])
# def denormalize(img: torch.Tensor):
#     deimg = img * log[:, None, None] + nu[:, None, None]
#     return deimg
# for n in range(self.dataset.N_CLASSES):
#     print_reconstructed_vs_true(
#         denormalize(torch.tensor(self.full_log['results']['0']['images'][n]['reconstruction'])),
#         denormalize(torch.tensor(self.full_log['results']['0']['images'][n]['original'])),
#         np.array([n]))
# for n in range(self.dataset.N_CLASSES):
#     print_reconstructed_vs_true(
#         torch.tensor(self.full_log['results']['0']['images'][n]['reconstruction']),
#         torch.tensor(self.full_log['results']['0']['images'][n]['original']),
#         np.array([n]))
