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

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super(VAE, self).__init__(args=args, dataset=dataset)
        self.E = get_encoder(args.dataset)(input_shape=dataset.INPUT_SHAPE,
                                           code_length=args.latent_space,
                                           variational=True)
        self.D = get_decoder(args.dataset)(code_length=args.latent_space,
                                           output_shape=dataset.INPUT_SHAPE)
        self.net = nn.Sequential(self.E, self.D).to(device=self.device)
        self.opt = self.Optimizer(self.net.parameters(), **self.optim_args)
        # self.E_opt = self.Optimizer(self.E.parameters(), **self.optim_args)
        # self.D_opt = self.Optimizer(self.D.parameters(), **self.optim_args)
        self.reconstruction_loss = nn.MSELoss()

    def kld_loss(self, mu: torch.Tensor, logvar: torch.Tensor):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld / self.args.batch_size

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        self.opt.zero_grad()
        # encoder forward in variational way
        encoder_out = self.E(x)
        latent_mu, latent_logvar = encoder_out[:x.shape[0], :], encoder_out[x.shape[0]:, :]

        # decoder forward
        latent_z = torch.mul(torch.randn_like(latent_mu), (0.5 * latent_logvar).exp()) + latent_mu
        outputs = self.D(latent_z)

        loss = self.reconstruction_loss(x, outputs) + self.args.kl_weight*self.kld_loss(latent_mu, latent_logvar)
        loss.backward()
        self.opt.step()
        return loss.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_out = self.E(x)
        latent_mu, _ = encoder_out[:x.shape[0], :], encoder_out[x.shape[0]:, :]
        # decoder forward
        outputs = self.D(latent_mu)
        return outputs
