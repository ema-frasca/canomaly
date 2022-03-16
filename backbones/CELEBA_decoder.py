from functools import reduce
from typing import Tuple

import torch
from torch import nn, mul
from torch.functional import F

from backbones.blocks_2d import UpsampleBlock, ResidualBlock
from backbones.utils.decoder import Decoder


class CELEBADecoder_old(nn.Module):

    def __init__(self, code_length: int, output_shape: Tuple[int, int, int]):
        super().__init__()

        c, h, w = output_shape
        # self.deepest_shape = (256, h // 8, w // 8)
        self.code_length = code_length
        self.d = 50

        self.scaler = h // 8

        self.fc1 = nn.Linear(code_length, self.d * self.scaler * self.scaler * self.scaler)

        self.dc1 = nn.ConvTranspose2d(self.d * self.scaler, self.d * 4, kernel_size=5, stride=2,
                                      padding=2, output_padding=1, bias=False)
        self.dc1_bn = nn.BatchNorm2d(self.d * 4)

        self.dc2 = nn.ConvTranspose2d(self.d * 4, self.d * 2, kernel_size=5, stride=2,
                                      padding=2, output_padding=1, bias=False)
        self.dc2_bn = nn.BatchNorm2d(self.d * 2)

        self.dc3 = nn.ConvTranspose2d(self.d * 2, self.d, kernel_size=5, stride=2,
                                      padding=2, output_padding=1, bias=False)
        self.dc3_bn = nn.BatchNorm2d(self.d)

        self.dc_out = nn.ConvTranspose2d(self.d, c, kernel_size=5, stride=1,
                                         padding=2, output_padding=0, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, self.d * self.scaler, self.scaler, self.scaler)
        x = self.dc1(x)
        x = F.leaky_relu(self.dc1_bn(x))
        x = self.dc2(x)
        x = F.leaky_relu(self.dc2_bn(x))
        x = self.dc3(x)
        x = F.leaky_relu(self.dc3_bn(x))
        x = self.dc_out(x)
        x = torch.sigmoid(x)

        return x


class CELEBADecoder(Decoder):
    """
    MNIST model decoder
    """

    def __init__(self, code_length: int, output_shape: Tuple[int, int, int]):
        """

        :param code_length:
        :param output_shape:
        """
        c, h, w = output_shape
        self.deepest_shape = (256, h // 8, w // 8)
        super(CELEBADecoder, self).__init__(code_length=code_length,
                                            output_shape=output_shape)

    def _set_conv_block(self):
        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=256, channel_out=128, activation_fn=self.activation_fn),
            UpsampleBlock(channel_in=128, channel_out=64, activation_fn=self.activation_fn),
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=self.activation_fn),
            ResidualBlock(channel_in=32, channel_out=32, activation_fn=self.activation_fn),
            nn.Conv2d(in_channels=32, out_channels=self.output_shape[0], kernel_size=1, bias=False)
        )

    def _set_fc_block(self):
        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.code_length, out_features=256),
            nn.BatchNorm1d(num_features=256),
            self.activation_fn,
            nn.Linear(in_features=256, out_features=reduce(mul, self.deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, self.deepest_shape)),
            self.activation_fn
        )
