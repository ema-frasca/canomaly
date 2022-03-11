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


class VAE_Module(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, forward_sample=False):
        super().__init__()
        self.E = encoder
        self.D = decoder
        self.forward_sample = forward_sample

    def sample(self, latent_mu: torch.Tensor, latent_logvar: torch.Tensor):
        return torch.mul(torch.randn_like(latent_mu), (0.5 * latent_logvar).exp()) + latent_mu

    def forward(self, x: torch.Tensor):
        """

        :param x:
        :return: reconstructed x, latent_mu, latent_logvar, latent_z
        """
        encoder_out = self.E(x)
        latent_mu, latent_logvar = encoder_out[:x.shape[0], :], encoder_out[x.shape[0]:, :]
        z = self.sample(latent_mu, latent_logvar)
        if self.forward_sample or self.training:
            recs = self.D(z)
        else:
            recs = self.D(latent_mu)

        return recs, latent_mu, latent_logvar, z


ModuleOuts = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# --dataset rec-fmnist --optim adam --lr 0.001 --scheduler_steps 2 --model rec-vae --kl_weight 1 --batch_size 64 --n_epochs 30 --latent_space 32 --approach joint
class RecVAE(ReconModel):
    NAME = 'rec-vae'
    VARIATIONAL = True
    CONDITIONAL = False

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        parser.add_argument('--latent_space', type=int, required=True,
                            help='Latent space dimensionality.')
        parser.add_argument('--kl_weight', type=float, required=True,
                            help='Weight for kl-divergence.')
        parser.add_argument('--forward_sample', action='store_true',
                            help='Set if you want to sample the decoder output.')
        parser.add_argument('--normalized_score', action='store_true',
                            help='Set if you want to normalize anomaly score in his components.')

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super(RecVAE, self).__init__(args=args, dataset=dataset)

        self.reconstruction_loss = lambda x, recs: F.mse_loss(recs, x, reduction='none')\
            .sum(dim=[i for i in range(1, len(x.shape))])\
            .mean()
        self.kld_not_reduction = lambda mu, logvar: -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def get_backbone(self):
        return VAE_Module(
            get_encoder(self.args.dataset)(input_shape=self.dataset.INPUT_SHAPE,
                                           code_length=self.args.latent_space,
                                           variational=self.VARIATIONAL,
                                           conditional=self.CONDITIONAL),
            get_decoder(self.args.dataset)(code_length=self.args.latent_space, output_shape=self.dataset.INPUT_SHAPE),
            forward_sample=self.args.forward_sample
        )

    def kld_loss(self, mu: torch.Tensor, logvar: torch.Tensor):
        kld = torch.mean(self.kld_not_reduction(mu, logvar), dim=0)
        return kld

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        self.opt.zero_grad()

        outputs, latent_mu, latent_logvar, z = self.forward(x, task)

        loss_reconstruction = self.reconstruction_loss(x, outputs)
        loss_kl = self.kld_loss(latent_mu, latent_logvar)
        loss = loss_reconstruction + self.args.kl_weight * loss_kl
        loss.backward()
        self.opt.step()
        if self.args.wandb:
            wandb.log({'rec_loss': loss_reconstruction, 'kl_loss': loss_kl*self.args.kl_weight})

        # logger.autolog_wandb(wandb_yes=self.args.wandb, locals=locals())
        return loss.item()

    def anomaly_score(self, recs: ModuleOuts, x: torch.Tensor) -> torch.Tensor:
        rec, mu, logvar, z = recs
        rec_loss = F.mse_loss(rec, x, reduction='none').mean(dim=[i for i in range(1, len(rec.shape))])
        kl_loss = self.kld_not_reduction(mu, logvar)
        if self.args.normalized_score:
            rec_loss_norm = ((rec_loss - self.rec_loss_train_stats[0]) /
                             (self.rec_loss_train_stats[1] - self.rec_loss_train_stats[0]))
            kl_loss_norm = ((kl_loss - self.kl_loss_train_stats[0]) /
                            (self.kl_loss_train_stats[1] - self.kl_loss_train_stats[0]))
            return rec_loss_norm + kl_loss_norm #todo: chiedi ad angel se devo pesare anche la kl per il beta di training
        else:
            return rec_loss + kl_loss

    def latents_from_outs(self, outs: ModuleOuts):
        rec, mu, logvar, z = outs
        return mu

    def train_on_task(self, task_loader: DataLoader, task: int):
        super(RecVAE, self).train_on_task(task_loader, task)
        self.net.eval()
        if self.args.normalized_score:
            kl_loss_train = []
            rec_loss_train = []
            for X, y in task_loader:
                X = X.to(self.device)
                recs, latent_mu, latent_logvar, z = self.forward(X)
                kl_l = self.kld_loss(latent_mu, latent_logvar)
                rec_l = self.reconstruction_loss(X, recs)
                kl_loss_train.append(kl_l.item())
                rec_loss_train.append(rec_l.item())

            # setattr(self, 'kl_loss_train_stats', (min(kl_loss_train), max(kl_loss_train)))
            # setattr(self, 'rec_loss_train_stats', (min(rec_loss_train), max(rec_loss_train)))
            self.kl_loss_train_stats = (min(kl_loss_train), max(kl_loss_train))
            self.rec_loss_train_stats = (min(rec_loss_train), max(rec_loss_train))

