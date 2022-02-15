from torch import nn
import torch
from argparse import Namespace, ArgumentParser

from models import CanomalyModel
from backbones import get_encoder, get_decoder
from datasets.utils.canomaly_dataset import CanomalyDataset


class VAE(CanomalyModel):
    NAME = 'vae'
    VARIATIONAL = False
    CONDITIONAL = False

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        parser.add_argument('--latent_space', type=int, required=True,
                            help='Latent space dimensionality.')
        parser.add_argument('--kl_weight', type=float, required=True,
                            help='Weight for kl-divergence.')
        parser.add_argument('--forward_sample', action='store_true',
                            help='Set if you want to sample the decoder output.')

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super(VAE, self).__init__(args=args, dataset=dataset)
        self.E = get_encoder(args.dataset)(input_shape=dataset.INPUT_SHAPE,
                                           code_length=args.latent_space,
                                           variational=True)
        self.D = get_decoder(args.dataset)(code_length=args.latent_space,
                                           output_shape=dataset.INPUT_SHAPE)
        self.net = nn.Sequential(self.E, self.D).to(device=self.device)
        self.opt = self.Optimizer(self.net.parameters(), **self.optim_args)
        self.reconstruction_loss = nn.MSELoss()

    def kld_loss(self, mu: torch.Tensor, logvar: torch.Tensor):
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        return kld

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        self.opt.zero_grad()

        # encoder forward in variational way
        encoder_out = self.E(x)
        latent_mu, latent_logvar = encoder_out[:x.shape[0], :], encoder_out[x.shape[0]:, :]

        # reparametrization trick
        latent_z = self.sample(latent_mu, latent_logvar)

        # decoder forward
        outputs = self.D(latent_z)

        loss = self.reconstruction_loss(x, outputs) + self.args.kl_weight*self.kld_loss(latent_mu, latent_logvar)
        loss.backward()
        self.opt.step()
        return loss.item()

    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        encoder_out = self.E(x)
        latent_mu, latent_logvar = encoder_out[:x.shape[0], :], encoder_out[x.shape[0]:, :]
        if self.args.forward_sample:
            z = self.sample(latent_mu, latent_logvar)
            return self.D(z)
        else:
            return self.D(latent_mu)

    def sample(self, latent_mu: torch.Tensor, latent_logvar: torch.Tensor):
        return torch.mul(torch.randn_like(latent_mu), (0.5 * latent_logvar).exp()) + latent_mu

