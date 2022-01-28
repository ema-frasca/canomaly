from torch import nn
import torch
from argparse import Namespace, ArgumentParser

from models import CanomalyModel
from backbones import get_encoder, get_decoder
from datasets.utils.canomaly_dataset import CanomalyDataset


class DAE(CanomalyModel):
    NAME = 'dae'
    VARIATIONAL = False
    CONDITIONAL = False

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        parser.add_argument('--latent_space', type=int, required=True,
                            help='Latent space dimensionality.')
        parser.add_argument('--noise_mean', type=float, default=0.,
                            help='Mean of the noise added.')
        parser.add_argument('--noise_std', type=float, default=0.5,
                            help='Standard deviation of the noise added.')

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super(DAE, self).__init__(args=args, dataset=dataset)
        self.E = get_encoder(args.dataset)(input_shape=dataset.INPUT_SHAPE,
                                           code_length=args.latent_space)
        self.D = get_decoder(args.dataset)(code_length=args.latent_space,
                                           output_shape=dataset.INPUT_SHAPE)
        self.net = nn.Sequential(self.E, self.D).to(device=self.device)
        self.opt = self.Optimizer(self.net.parameters(), **self.optim_args)
        self.loss = nn.MSELoss()

    def add_noise(self, imgs: torch.Tensor):
        return imgs + torch.randn_like(imgs) * self.args.noise_std + self.args.noise_mean

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        self.opt.zero_grad()
        x_noised = self.add_noise(x)
        outputs = self.forward(x_noised)
        loss = self.loss(outputs, x)
        loss.backward()
        self.opt.step()
        return loss.item()
