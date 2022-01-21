from functools import reduce
from typing import Tuple

from torch import nn, mul

from backbones.blocks_2d import UpsampleBlock, ResidualBlock
from backbones.utils.decoder import Decoder


class CIFAR10Decoder(Decoder):
    """
    MNIST model decoder
    """

    def __init__(self, code_length: int, deepest_shape: Tuple[int, int, int],
                 output_shape: Tuple[int, int, int]):
        """

        :param code_length:
        :param deepest_shape:
        :param output_shape:
        """
        super(CIFAR10Decoder, self).__init__(deepest_shape=deepest_shape, code_length=code_length,
                                             output_shape=output_shape)

    def _set_conv_block(self):
        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=256, channel_out=128, activation_fn=self.activation_fn),
            UpsampleBlock(channel_in=128, channel_out=64, activation_fn=self.activation_fn),
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=self.activation_fn),
            ResidualBlock(channel_in=32, channel_out=32, activation_fn=self.activation_fn),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, bias=False)
        )

    def _set_fc_block(self):
        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=256),
            nn.BatchNorm1d(num_features=256),
            self.activation_fn,
            nn.Linear(in_features=256, out_features=reduce(mul, self.deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, self.deepest_shape)),
            self.activation_fn
        )
