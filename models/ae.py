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
        parser.add_argument('--splits', action='store_true', help='Joint model with a net per task.')

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super(AE, self).__init__(args=args, dataset=dataset)
        self.E = get_encoder(args.dataset)(input_shape=dataset.INPUT_SHAPE, code_length=args.latent_space)
        self.D = get_decoder(args.dataset)(code_length=args.latent_space, output_shape=dataset.INPUT_SHAPE)
        self.net = nn.Sequential(self.E, self.D).to(device=self.device)

        if self.args.splits:
            self.net = nn.ModuleList([nn.Sequential(
                get_encoder(args.dataset)(input_shape=dataset.INPUT_SHAPE, code_length=args.latent_space),
                get_decoder(args.dataset)(code_length=args.latent_space, output_shape=dataset.INPUT_SHAPE)
            ) for i in range(self.dataset.n_tasks)]).to(device=self.device)

        self.opt = self.Optimizer(self.net.parameters(), **self.optim_args)
        self.loss = nn.MSELoss()

        self.batch_loss = nn.MSELoss(reduction='none')
        self.last_task = 0

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        self.last_task = task

        self.opt.zero_grad()
        outputs = self.forward(x, task)
        loss = self.loss(outputs, x)
        loss.backward()
        self.opt.step()
        return loss.item()

    def forward(self, x: torch.Tensor, task: int = None):
        if self.args.splits:
            if task is not None:
                return self.net[task](x)

            if self.last_task == 0:
                return self.net[0](x)

            outputs = torch.stack([net(x) for net in self.net[:self.last_task+1]])
            losses = torch.stack([self.batch_loss(out, x).mean(dim=(1, 2, 3)) for out in outputs])
            min_idx = losses.argmin(dim=0)
            # best_outs = torch.stack([outputs[min_idx[i], i] for i in range(outputs.shape[1])])
            return outputs[min_idx, torch.arange(outputs.shape[1])]
        else:
            return self.net(x)

