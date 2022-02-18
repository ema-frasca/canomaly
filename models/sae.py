from torch import nn
import torch
from argparse import Namespace, ArgumentParser

from models import CanomalyModel
from backbones import get_encoder, get_decoder
from datasets.utils.canomaly_dataset import CanomalyDataset


class SAE_Module(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.E = encoder
        self.D = decoder

    def forward(self, x: torch.Tensor):
        z = self.E(x)
        recs = self.D(z)
        if self.training:
            return recs, z
        return recs


class SAE(CanomalyModel):
    NAME = 'sae'
    VARIATIONAL = False
    CONDITIONAL = False

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        parser.add_argument('--latent_space', type=int, required=True,
                            help='Latent space dimensionality.')
        parser.add_argument('--sparse_weight', type=float, required=True,
                            help='Weight for sparse loss.')
        parser.add_argument('--norm_order', type=int, default=2, help='Normalization order',
                            choices=[1, 2])

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super(SAE, self).__init__(args=args, dataset=dataset)

        self.reconstruction_loss = nn.MSELoss()

    def get_backbone(self):
        return SAE_Module(
            get_encoder(self.args.dataset)(input_shape=self.dataset.INPUT_SHAPE, code_length=self.args.latent_space),
            get_decoder(self.args.dataset)(code_length=self.args.latent_space, output_shape=self.dataset.INPUT_SHAPE),
        )

    def sparse_loss(self, z: torch.Tensor):
        # mean to normalize on batch size
        return torch.linalg.vector_norm(z, ord=self.args.norm_order, dim=1).mean()

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        self.opt.zero_grad()
        outputs, z = self.forward(x, task)
        loss = self.reconstruction_loss(outputs, x) + self.args.sparse_weight * self.sparse_loss(z)
        loss.backward()
        self.opt.step()
        return loss.item()
