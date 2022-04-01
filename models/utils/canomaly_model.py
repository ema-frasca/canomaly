from typing import Union
import torch
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR
from abc import abstractmethod
from argparse import Namespace, ArgumentParser
from utils.config import config
from datasets.utils.canomaly_dataset import CanomalyDataset
from utils.optims import get_optim
from utils.writer import writer
from utils.logger import logger
from utils.metrics import (reconstruction_error, compute_exp_metrics, reconstruction_confusion_matrix, compute_task_auc,
                           print_reconstructed_vs_true)
from typing import Union
from random import random
import wandb


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
        self.joint = args.approach == 'joint'
        self.splits = args.approach == 'splits'

        Optim, optim_args = get_optim(args)
        self.Optimizer = Optim
        self.optim_args = optim_args
        self.full_log = {**vars(args), 'results': {}, 'knowledge': {}}

        if self.splits:
            self.net = nn.ModuleList([self.get_backbone() for i in range(self.dataset.n_tasks)]).to(device=self.device)
        else:
            self.net = self.get_backbone().to(device=self.device)

        self.cur_task = 0

        self.opt = self.get_opt()
        self.scheduler = self.get_scheduler()

    @abstractmethod
    def get_backbone(self) -> torch.nn.Module:
        pass

    def get_opt(self) -> Optimizer:
        return self.Optimizer(self.net.parameters(), **self.optim_args)

    def get_scheduler(self) -> Union[StepLR, None]:
        if self.args.scheduler_steps > 0 and self.args.n_epochs > 1:
            return StepLR(self.opt, step_size=(self.args.n_epochs // (self.args.scheduler_steps+1)), gamma=self.args.scheduler_gamma)
        else:
            return None

    @abstractmethod
    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int) -> float:
        pass

    def anomaly_score(self, recs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recs, x, reduction='none').mean(dim=[i for i in range(len(recs.shape))][1:])

    def forward(self, x: torch.Tensor, task: int = None, **kwargs) -> torch.Tensor:
        if self.splits:
            if task is not None:
                return self.net[task](x, **kwargs)
            if self.cur_task == 0:
                return self.net[0](x, **kwargs)

            outputs = torch.stack([net(x, **kwargs) for net in self.net[:self.cur_task+1]])
            losses = torch.stack([self.anomaly_score(out, x) for out in outputs])
            min_idx = losses.argmin(dim=0)
            return outputs[min_idx, torch.arange(outputs.shape[1])]
        else:
            return self.net(x, **kwargs)

    def net_train(self):
        self.net.train()

    def net_eval(self):
        self.net.eval()

    @torch.no_grad()
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

            # from utils.metrics import print_reconstructed_vs_true
            # import numpy as np
            # n = 0
            # nu, log = torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5]) # celeba
            # nu, log = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]) # mvtec
            # def denormalize(img: torch.Tensor):
            #     deimg = img * log[:, None, None] + nu[:, None, None]
            #     return deimg
            # print_reconstructed_vs_true(denormalize(X[n].cpu().detach()), denormalize(outs[n].cpu().detach()), np.array([n]))

            self.full_log['results'][str(task)]['rec_errs'].extend(rec_errs.tolist())
            if len(images_sample) < self.dataset.N_CLASSES and random() < 0.1:
                for i in range(len(y)):
                    if str(y[i].item()) not in images_sample:
                        images_sample[str(y[i].item())] = {'original': X[i].tolist(),
                                                           'reconstruction': outs[i].tolist()}
                        # print_reconstructed_vs_true(outs[i], X[i], y[i], (28, 28))
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
            # if self.args.wandb:
            #     wandb.log({"epoch": e, "lr": self.scheduler.get_last_lr()[0]})
            keep_progress = True  # if e == self.args.n_epochs - 1 else False
            progress = logger.get_tqdm(task_loader,
                                       f'TRAIN on task {task+1}/{self.dataset.n_tasks} - epoch {e+1}/{self.args.n_epochs}',
                                       leave=keep_progress)
            for x, y in progress:
                loss = self.train_on_batch(x.to(self.device), y.to(self.device), task)
                progress.set_postfix({'loss': loss})
                if self.args.wandb:
                    wandb.log({"loss": loss, "epoch": e, "lr": self.scheduler.get_last_lr()[0], "task": task})
            self.validate()
            self.scheduler_step()

    @torch.no_grad()
    def validate(self):
        if self.args.wandb:
            if self.args.dataset == 'mvtecad':
                ok = True
                while ok:
                    X, y = next(iter(self.dataset.test_loader()))
                    if (y==0).sum() > 0 and (y==1).sum() > 0:
                        ok = False
                X = X.to(self.device)
                recs = self.forward(X)
                good, bad = X[y == 0][0], X[y == 1][0]
                good_rec, bad_rec = recs[y == 0][0], recs[y == 1][0]
                orig = torch.cat((good, bad), dim=2)
                recs = torch.cat((good_rec, bad_rec), dim=2)
                full = torch.cat((orig, recs), dim=1).permute(1, 2, 0)
                img = wandb.Image(full.cpu().numpy(), caption='top: origins; bottom: reconstructions')
                wandb.log({'imgs': img})

    def scheduler_step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def train_on_dataset(self):
        logger.log(vars(self.args))
        # logger.log(self.net)
        loader = self.dataset.joint_loader if self.joint else self.dataset.task_loader
        freezed_params = [param.data.clone() for param in self.net.parameters()] if self.joint else []
        for i, task_dl in enumerate(loader()):
            self.cur_task = i
            if self.joint:
                for pidx, param in enumerate(self.net.parameters()):
                    param.data = freezed_params[pidx].clone()
            self.opt = self.get_opt()
            self.scheduler = self.get_scheduler()
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

        # from utils.metrics import print_reconstructed_vs_true
        # import numpy as np
        # n = 0
        # nu, log = torch.tensor([0.4914, 0.4822, 0.4465]), torch.tensor([0.2470, 0.2435, 0.2615])
        # # grayscale:
        # nu, log = torch.tensor([(0.4914+0.4822+0.4465)/3]), torch.tensor([(0.2470+0.2435+0.2615)/3])
        # def denormalize(img: torch.Tensor):
        #     deimg = img * log[:, None, None] + nu[:, None, None]
        #     return deimg
        # for n in range(self.dataset.N_CLASSES):
        #     print_reconstructed_vs_true(
        #         denormalize(torch.tensor(self.full_log['results']['0']['images'][n]['reconstruction'])),
        #         denormalize(torch.tensor(self.full_log['results']['0']['images'][n]['original'])),
        #         np.array([n]))

        writer.write_log(res_log, result=True)
