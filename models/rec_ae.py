from typing import Tuple

from torch import nn
import torch
from argparse import Namespace, ArgumentParser

from models.utils.recon_model import ReconModel
from backbones import get_encoder, get_decoder
from datasets.utils.canomaly_dataset import CanomalyDataset


class AE_Module(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.E = encoder
        self.D = decoder

    def forward(self, x: torch.Tensor):
        latents = self.E(x)
        recs = self.D(latents)
        return recs, latents


ModuleOuts = Tuple[torch.Tensor, torch.Tensor]


class RecAE(ReconModel):
    NAME = 'rec-ae'
    VARIATIONAL = False
    CONDITIONAL = False

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        parser.add_argument('--latent_space', type=int, required=True, help='Latent space dimensionality.')

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super(RecAE, self).__init__(args=args, dataset=dataset)

        self.loss = nn.MSELoss()

    def latents_from_outs(self, outs: ModuleOuts) -> torch.Tensor:
        return outs[1]

    def get_backbone(self):
        return AE_Module(
            get_encoder(self.args.dataset)(input_shape=self.dataset.INPUT_SHAPE, code_length=self.args.latent_space),
            get_decoder(self.args.dataset)(code_length=self.args.latent_space, output_shape=self.dataset.INPUT_SHAPE),
        )

    def anomaly_score(self, recs: ModuleOuts, x: torch.Tensor):
        return super().anomaly_score(recs[0], x)

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        self.opt.zero_grad()
        outputs, latents = self.forward(x, task)
        loss = self.loss(outputs, x)
        loss.backward()
        self.opt.step()
        return loss.item()
