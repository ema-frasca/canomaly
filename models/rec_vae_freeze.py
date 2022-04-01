from typing import Tuple, List
from torch import nn
import torch
from argparse import Namespace, ArgumentParser

from torch.utils.data import DataLoader

from models import CanomalyModel
from backbones import get_encoder, get_decoder
from datasets.utils.canomaly_dataset import CanomalyDataset
from models.utils.recon_model import ReconModel
from utils.logger import logger
from torch.functional import F
import wandb
from models.rec_vae import RecVAE


# --dataset rec-fmnist --optim adam --lr 0.001 --scheduler_steps 2 --model rec-vae --kl_weight 1 --batch_size 64 --n_epochs 30 --latent_space 32 --approach joint
class RecVaeFreezE(RecVAE):
    NAME = 'rec-vae-freeze'
    VARIATIONAL = True
    CONDITIONAL = False

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        assert args.approach != 'splits', 'splits approach not implemented for this model... sorry :('

        super().__init__(args=args, dataset=dataset)

    def get_opt(self):
        if self.cur_task == 0:
            return self.Optimizer(self.net.parameters(), **self.optim_args)
        else:
            return self.Optimizer(self.net.D.parameters(), **self.optim_args)

    def net_train(self):
        if self.cur_task == 0:
            self.net.train()
        else:
            self.net.D.train()

    def train_on_dataset(self):
        logger.log(vars(self.args))
        # logger.log(self.net)
        loader = self.dataset.joint_loader if self.joint else self.dataset.task_loader
        for i, task_dl in enumerate(loader()):
            self.cur_task = i
            self.opt = self.get_opt()
            self.scheduler = self.get_scheduler()
            self.train_on_task(task_dl, i)
            self.full_log['knowledge'][str(i)] = self.dataset.last_seen_classes.copy()
            # evaluate on test
            self.test_step(self.dataset.test_loader(), i)
