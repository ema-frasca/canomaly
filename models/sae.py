from torch import nn
import torch
from argparse import Namespace, ArgumentParser

from models import CanomalyModel
from backbones import get_encoder, get_decoder
from datasets.utils.canomaly_dataset import CanomalyDataset


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
        parser.add_argument('--norm_order', type=int, required=True, help='Normalization order',
                            choices=[1, 2])

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super(SAE, self).__init__(args=args, dataset=dataset)
        self.E = get_encoder(args.dataset)(input_shape=dataset.INPUT_SHAPE,
                                           code_length=args.latent_space)
        self.D = get_decoder(args.dataset)(code_length=args.latent_space,
                                           output_shape=dataset.INPUT_SHAPE)
        self.net = nn.Sequential(self.E, self.D).to(device=self.device)
        self.opt = self.Optimizer(self.net.parameters(), **self.optim_args)
        self.reconstruction_loss = nn.MSELoss()

    def sparse_loss(self, z: torch.Tensor):
        # mean to normalize on batch size
        return torch.linalg.vector_norm(z, ord=self.args.norm_order, dim=1).mean()

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        self.opt.zero_grad()
        z = self.E(x)
        outputs = self.D(z)
        loss = self.reconstruction_loss(outputs, x) + self.args.sparse_weight * self.sparse_loss(z)
        loss.backward()
        self.opt.step()
        return loss.item()
