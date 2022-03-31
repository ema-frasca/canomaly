import torch
from argparse import Namespace, ArgumentParser
from torch.functional import F
from torch.utils.data import DataLoader

from datasets.utils.canomaly_dataset import CanomalyDataset
import wandb
from models.rec_vae import RecVAE
from continual.buffer import Buffer
from models.utils.recon_model import ReconModel
from utils.logger import logger


# --dataset rec-fmnist --optim adam --lr 0.001 --scheduler_steps 2 --model rec-vae-ewc --batch_size 64 --n_epochs 30 --latent_space 32 --kl_weight 1 --approach continual --e_lambda 1 --gamma 1 --wandb --logs
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

        self.fisher = Fisher(self)

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
                {'rec_loss': loss_reconstruction, 'kl_loss': loss_kl * self.args.kl_weight, 'fisher_loss':
                    penalty, 'fisher_loss_weighted': self.args.e_lambda * penalty})

        # logger.autolog_wandb(wandb_yes=self.args.wandb, locals=locals())
        return loss.item()

    def train_on_task(self, task_loader: DataLoader, task: int):
        super().train_on_task(task_loader, task)
        self.fisher.update_fish(task_loader)


class Fisher:
    def __init__(self, model: ReconModel):
        self.checkpoint = None
        self.fish = None
        self.model = model

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument('--gamma', type=float, required=True,
                            help='gamma parameter for EWC online')

    def penalty(self) -> torch.Tensor:
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.model.config.device)
        else:

            penalty = (self.fish * ((self.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def update_fish(self, dl: DataLoader):
        fish = torch.zeros_like(self.get_params())

        for j, data in enumerate(dl):
            inputs, labels = data
            inputs, labels = inputs.to(self.model.config.device), labels.to(self.model.config.device)
            for ex, lab in zip(inputs, labels):
                self.model.opt.zero_grad()
                output = self.model.forward(ex.unsqueeze(0))
                rec_loss = self.model.reconstruction_loss(output[0],
                                                          ex.unsqueeze(0))
                kld_loss = self.model.kld_not_reduction(*output[1:3])
                loss = rec_loss + self.model.args.kl_weight * kld_loss
                detached_loss = torch.mean(loss.detach().clone())
                loss.backward()

                fish += detached_loss * self.get_grads() ** 2

        fish /= (len(dl) * dl.batch_size)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.model.args.gamma
            self.fish += fish

        self.checkpoint = self.get_params().data.clone()

    def get_params(self):
        params = []
        for pp in list(self.model.net.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self):
        grads = []
        for pp in list(self.model.net.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)
