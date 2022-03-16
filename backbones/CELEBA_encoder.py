from typing import Tuple

import torch
from torch import nn
from torch.functional import F

from backbones.blocks_2d import DownsampleBlock, ResidualBlock
from backbones.utils.encoder import Encoder


class CELEBAEncoder_old(nn.Module):

    def __init__(self, input_shape: Tuple[int, int, int], code_length: int, variational: bool = False):
        super().__init__()

        c, h, w = input_shape
        # self.deepest_shape = (256, h // 8, w // 8)
        self.code_length = code_length
        self.variational = variational
        self.d = 50

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.d, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(self.d)

        self.conv2 = nn.Conv2d(self.d, self.d * 2, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(self.d * 2)

        self.conv3 = nn.Conv2d(self.d * 2, self.d * 4, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(self.d * 4)

        self.conv4 = nn.Conv2d(self.d * 4, self.d * 4, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn_4 = nn.BatchNorm2d(self.d * 4)
        if h == 64:
            self.conv_out_size = 3
        elif h == 128:
            self.conv_out_size = 7
        self.fc = nn.Linear(self.d * 4 * self.conv_out_size * self.conv_out_size, self.d * 4)

        self.linear_means = nn.Linear(self.d * 4, code_length)

        if variational:
            self.linear_log_var = nn.Linear(self.d * 4, code_length)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.leaky_relu(self.bn_1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn_2(x))
        x = self.conv3(x)
        x = F.leaky_relu(self.bn_3(x))
        x = self.conv4(x)
        x = F.leaky_relu(self.bn_4(x))
        x = x.view([-1, self.d * 4 * self.conv_out_size * self.conv_out_size])

        x = F.leaky_relu(self.fc(x))
        means = self.linear_means(x)
        if not self.variational:
            return means

        log_vars = self.linear_log_var(x)
        return torch.cat([means, log_vars])


class CELEBAEncoder(Encoder):
    """
    MNIST model encoder.
    """

    def __init__(self, input_shape: Tuple[int, int, int], code_length: int, variational: bool = False,
                 conditional: bool = False) -> None:
        """
        Class constructor:
        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        """
        c, h, w = input_shape
        self.deepest_shape = (256, h // 8, w // 8)
        super(CELEBAEncoder, self).__init__(input_shape=input_shape, code_length=code_length, variational=variational,
                                             conditional=conditional)

    def _set_conv_block(self):
        c, h, w = self.input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, bias=False),
            self.activation_fn,
            ResidualBlock(channel_in=32, channel_out=32, activation_fn=self.activation_fn),
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=self.activation_fn),
            DownsampleBlock(channel_in=64, channel_out=128, activation_fn=self.activation_fn),
            DownsampleBlock(channel_in=128, channel_out=256, activation_fn=self.activation_fn),
        )