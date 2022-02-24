from typing import Tuple, Union

from torch import nn
import torch
from argparse import Namespace, ArgumentParser

from models import CanomalyModel
from backbones import get_encoder, get_decoder
from datasets.utils.canomaly_dataset import CanomalyDataset
from torch.nn import functional as F


class VQVAE_Module(nn.Module):
    def __init__(self, encoder: nn.Module, vq_layer: nn.Module, decoder: nn.Module):
        super().__init__()
        self.E = encoder
        self.vq_layer = vq_layer
        self.D = decoder

    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, float], torch.Tensor]:
        encoder_out = self.E(x)
        quantized_inputs, vq_loss = self.vq_layer(encoder_out)
        if self.training:
            return self.D(quantized_inputs), vq_loss
        else:
            return self.D(quantized_inputs)


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, float]:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = (torch.sum(flat_latents ** 2, dim=1, keepdim=True) +
                torch.sum(self.embedding.weight ** 2, dim=1) -
                2 * torch.matmul(flat_latents, self.embedding.weight.t()))  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]


class VQVAE(CanomalyModel):
    NAME = 'vqvae'
    VARIATIONAL = False
    CONDITIONAL = False

    @staticmethod
    def add_model_args(parser: ArgumentParser):
        parser.add_argument('--latent_space', type=int, required=True,
                            help='Latent space dimensionality.')
        parser.add_argument('--num_embeddings', type=int, help='Number of embedding')
        parser.add_argument('--beta', type=float, help='Weight in vector quantization.')

    def __init__(self, args: Namespace, dataset: CanomalyDataset):
        super(VQVAE, self).__init__(args=args, dataset=dataset)
        self.reconstruction_loss = nn.MSELoss()

    def get_backbone(self):
        return VQVAE_Module(
            get_encoder(self.args.dataset)(input_shape=self.dataset.INPUT_SHAPE,
                                           code_length=self.args.latent_space).conv,
            VectorQuantizer(self.args.num_embeddings,
                            self.args.latent_space,
                            self.args.beta),
            get_decoder(self.args.dataset)(code_length=self.args.latent_space,
                                           output_shape=self.dataset.INPUT_SHAPE).conv,
        )

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor, task: int):
        self.opt.zero_grad()
        outputs, vq_loss = self.forward(x)
        loss = self.reconstruction_loss(x, outputs) + vq_loss
        loss.backward()
        self.opt.step()
        return loss.item()
