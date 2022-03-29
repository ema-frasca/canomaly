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
from models.rec_ae import RecAE
from continual.buffer import Buffer


class RecAEER(RecAE):
    NAME = 'rec-ae-er'
    VARIATIONAL = True
    CONDITIONAL = False

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        RecAE.add_model_args(parser)
        Buffer.add_args(parser)

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super().__init__(args=args, dataset=dataset)

        self.buffer = Buffer(self.args, self.config.device)

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        real_batch_size = x.shape[0]
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data()
            x = torch.cat((x, buf_inputs))
            y = torch.cat((y, buf_labels))

        # todo: add not augmented data in buffer
        self.buffer.add_data(examples=x[:real_batch_size], labels=y[:real_batch_size])

        return super().train_on_batch(x, y, task)