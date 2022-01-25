from torch import nn
import torch
from argparse import Namespace, ArgumentParser

from models import CanomalyModel
from backbones import get_encoder, get_decoder
from datasets.utils.canomaly_dataset import CanomalyDataset


class AE(CanomalyModel):
    NAME = 'AE'
    VARIATIONAL = False
    CONDITIONAL = False

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        parser.add_argument('--latent_space', type=int, required=True,
                            help='Latent space dimensionality.')

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super(AE, self).__init__(args=args, dataset=dataset)
        self.E = get_encoder(args.dataset)(input_shape=dataset.INPUT_SHAPE,
                                           code_length=args.latent_space)
        self.D = get_decoder(args.dataset)(code_length=args.latent_space,
                                           output_shape=dataset.INPUT_SHAPE)
        self.net = nn.Sequential(self.E, self.D)
        self.opt = self.Optimizer(self.net.parameters(), **self.optim_args)
        self.loss = nn.MSELoss()

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        self.opt.zero_grad()
        outputs = self.forward(x)
        loss = self.loss(outputs, x)
        loss.backward()
        self.opt.step()
        return loss.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
