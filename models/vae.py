from torch import nn
import torch
from argparse import Namespace, ArgumentParser

from models import CanomalyModel
from backbones import get_encoder, get_decoder
from datasets.utils.canomaly_dataset import CanomalyDataset


class VAE(CanomalyModel):
    NAME = 'VAE'
    VARIATIONAL = False
    CONDITIONAL = False

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        parser.add_argument('--latent_space', type=int, required=True,
                            help='Latent space dimensionality.')
        parser.add_argument('--beta_kl', type=float, required=True,
                            help='Weight for kldivergence.')

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super(VAE, self).__init__(args=args, dataset=dataset)
        self.E = get_encoder(args.dataset)(input_shape=dataset.INPUT_SHAPE,
                                           code_length=args.latent_space,
                                           variational=True)
        self.D = get_decoder(args.dataset)(code_length=args.latent_space,
                                           output_shape=dataset.INPUT_SHAPE)
        self.net = nn.Sequential(self.E, self.D)
        self.opt = self.Optimizer(self.net.parameters(), **self.optim_args)
        # self.E_opt = self.Optimizer(self.E.parameters(), **self.optim_args)
        # self.D_opt = self.Optimizer(self.D.parameters(), **self.optim_args)
        self.loss = nn.MSELoss()

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        self.opt.zero_grad()
        # encoder forward in variational way
        encoder_out = self.net[0](x)
        latent_mu, latent_logvar = encoder_out[:x.shape[0], :], encoder_out[x.shape[0]:, :]
        latent_z = latent_sample(latent_mu, latent_logvar)
        # decoder forward
        outputs = self.net[1](latent_z)
        loss = self.loss(x, outputs) + self.args.beta_kl*kldivergence(latent_mu, latent_logvar)

        loss.backward()
        self.opt.step()
        return loss.item()

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        encoder_out = self.net[0](x)
        latent_mu, _ = encoder_out[:x.shape[0], :], encoder_out[x.shape[0]:, :]
        # decoder forward
        outputs = self.net[1](latent_mu)
        return outputs


def latent_sample(mu: torch.Tensor, logvar: torch.Tensor):
    # the reparameterization trick
    std = logvar.mul(0.5).exp_()
    eps = torch.empty_like(std).normal_()
    return eps.mul(std).add_(mu)


def kldivergence(mu: torch.Tensor, logvar: torch.Tensor):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
