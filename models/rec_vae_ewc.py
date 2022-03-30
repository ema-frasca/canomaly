import torch
from argparse import Namespace, ArgumentParser
from torch.functional import F
from torch.utils.data import DataLoader

from datasets.utils.canomaly_dataset import CanomalyDataset
import wandb
from models.rec_vae import RecVAE
from continual.buffer import Buffer


# --dataset rec-fmnist --optim adam --lr 0.001 --scheduler_steps 2 --model rec-vae --kl_weight 1 --batch_size 64 --n_epochs 30 --latent_space 32 --approach joint
class RecVAEEWC(RecVAE):
    NAME = 'rec-vae-ewc'
    VARIATIONAL = True
    CONDITIONAL = False

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        RecVAE.add_model_args(parser)
        Fisher.add_args(parser)
        parser.add_argument('--e_lambda', type=float, required=True,
                            help='lambda weight for EWC')

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super().__init__(args=args, dataset=dataset)

        self.fisher = Fisher(self.net, self.opt, self.config)

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        self.opt.zero_grad()

        outputs, latent_mu, latent_logvar, z = self.forward(x, task)
        penalty = self.fisher.penalty()

        loss_reconstruction = self.reconstruction_loss(x, outputs)
        loss_kl = self.kld_loss(latent_mu, latent_logvar)
        loss = loss_reconstruction + self.args.kl_weight * loss_kl + self.args.e_lambda * penalty
        loss.backward()
        self.opt.step()
        if self.args.wandb:
            wandb.log(
                {'rec_loss': loss_reconstruction, 'kl_loss': loss_kl * self.args.kl_weight, 'fisher_loss': penalty})

        # logger.autolog_wandb(wandb_yes=self.args.wandb, locals=locals())
        return loss.item()

    def train_on_task(self, task_loader: DataLoader, task: int):
        super().train_on_task(task_loader)


class Fisher:
    def __init__(self, net, opt, config):
        self.checkpoint = None
        self.fish = None
        self.net = net
        self.opt = opt
        self.config = config

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument('--gamma', type=float, required=True,
                            help='gamma parameter for EWC online')

    def penalty(self) -> torch.Tensor:
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.config.device)
        else:
            penalty = (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def update_fish(self, dataset):
        fish = torch.zeros_like(self.net.get_params())

        for j, data in enumerate(dataset.train_loader):
            inputs, labels, _ = data
            inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                loss = F.mse_loss(output, lab.unsqueeze(0),
                                  reduction='none')
                detached_loss = torch.mean(loss.detach().clone())
                loss = torch.mean(loss)
                loss.backward()
                fish += detached_loss * self.net.get_grads() ** 2

        fish /= (len(dataset.train_loader) * self.net.args.batch_size)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.net.args.gamma
            self.fish += fish

        self.checkpoint = self.net.get_params().data.clone()
