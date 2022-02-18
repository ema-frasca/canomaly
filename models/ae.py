from torch import nn
import torch
from argparse import Namespace, ArgumentParser

from models import CanomalyModel
from backbones import get_encoder, get_decoder
from datasets.utils.canomaly_dataset import CanomalyDataset


class AE(CanomalyModel):
    NAME = 'ae'
    VARIATIONAL = False
    CONDITIONAL = False

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        parser.add_argument('--latent_space', type=int, required=True, help='Latent space dimensionality.')

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super(AE, self).__init__(args=args, dataset=dataset)

        self.loss = nn.MSELoss()

    def get_backbone(self):
        return nn.Sequential(
            get_encoder(self.args.dataset)(input_shape=self.dataset.INPUT_SHAPE, code_length=self.args.latent_space),
            get_decoder(self.args.dataset)(code_length=self.args.latent_space, output_shape=self.dataset.INPUT_SHAPE),
        )

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        self.opt.zero_grad()
        outputs = self.forward(x, task)
        loss = self.loss(outputs, x)
        loss.backward()
        self.opt.step()
        return loss.item()
