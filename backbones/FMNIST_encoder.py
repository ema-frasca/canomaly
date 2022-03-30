from typing import Tuple
from torch.functional import F
import torch
from torch import nn
from einops import rearrange
from backbones.blocks_2d import DownsampleBlock
from backbones.utils.encoder import Encoder


class FMNISTEncoder(Encoder):
    """
    FMNIST model encoder.
    """

    def __init__(self, input_shape: Tuple[int, int, int], code_length: int, variational: bool = False,
                 conditional: bool = False) -> None:
        """
        Class constructor:
        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        """
        c, h, w = input_shape
        self.deepest_shape = (64, h // 4, w // 4)

        super().__init__(input_shape=input_shape, code_length=code_length, variational=variational,
                         conditional=conditional)

    def _set_conv_block(self):
        c, h, w = self.input_shape

        self.conv = nn.Sequential(
            DownsampleBlock(channel_in=c, channel_out=32, activation_fn=self.activation_fn),
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=self.activation_fn),
        )


class FMNISTEncoder_old(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], code_length: int, variational: bool = False,
                 conditional: bool = False) -> None:
        """
        Class constructor:
        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        """

        super().__init__()
        c, h, w = input_shape
        # self.deepest_shape = (256, h // 8, w // 8)
        self.code_length = code_length
        self.variational = variational
        self.d = 10

        self.lin_1 = nn.Linear(c*h*w, self.d*2)
        self.bn_1 = nn.BatchNorm1d(self.d*2)

        self.lin_2 = nn.Linear(self.d*2, self.d)
        self.bn_2 = nn.BatchNorm1d(self.d)

        self.linear_means = nn.Linear(self.d, code_length)

        if variational:
            self.linear_log_var = nn.Linear(self.d, code_length)

    def forward(self, x: torch.Tensor):
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.lin_1(x)
        x = F.leaky_relu(self.bn_1(x))
        x = self.lin_2(x)
        x = F.leaky_relu(self.bn_2(x))
        means = self.linear_means(x)
        if not self.variational:
            return means
        log_vars = self.linear_log_var(x)
        return torch.cat([means, log_vars])

