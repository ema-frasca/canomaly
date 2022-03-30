from functools import reduce
from typing import Tuple
from einops import rearrange
import torch
from torch.functional import F
from torch import nn, mul

from backbones.blocks_2d import UpsampleBlock
from backbones.utils.decoder import Decoder


class FMNISTDecoder(Decoder):
    """
    FMNIST model decoder
    """

    def __init__(self, code_length: int, output_shape: Tuple[int, int, int]):
        """

        :param code_length:
        :param output_shape:
        """
        c, h, w = output_shape
        self.deepest_shape = (64, h // 4, w // 4)
        super().__init__(code_length=code_length,
                         output_shape=output_shape)

    def _set_conv_block(self):
        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=self.activation_fn),
            UpsampleBlock(channel_in=32, channel_out=16, activation_fn=self.activation_fn),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=False)
        )

    def _set_fc_block(self):
        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.code_length, out_features=64),
            nn.BatchNorm1d(num_features=64),
            self.activation_fn,
            nn.Linear(in_features=64, out_features=reduce(mul, self.deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, self.deepest_shape)),
            self.activation_fn
        )


class FMNISTDecoder_old(nn.Module):

    def __init__(self, code_length: int, output_shape: Tuple[int, int, int]):
        super().__init__()

        c, h, w = output_shape
        self.c = c
        self.h = h
        self.w = w
        self.code_length = code_length
        self.d = 10
        self.lin_1 = nn.Linear(self.code_length, self.d)
        self.bn_1 = nn.BatchNorm1d(self.d)

        self.lin_2 = nn.Linear(self.d, self.d*2)
        self.bn_2 = nn.BatchNorm1d(self.d*2)
        self.fc = nn.Linear(self.d*2, c*h*w)

    def forward(self, x):
        x = self.lin_1(x)
        x = F.leaky_relu(self.bn_1(x))
        x = self.lin_2(x)
        x = F.leaky_relu(self.bn_2(x))
        x = self.fc(x)
        x = torch.sigmoid(x)

        return rearrange(x, 'b (c h w) -> b c h w', h=self.h, c=self.c, w=self.w)
