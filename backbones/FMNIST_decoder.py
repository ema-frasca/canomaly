from functools import reduce
from typing import Tuple

from torch import nn, mul

from backbones.blocks_2d import UpsampleBlock
from backbones.utils.decoder import Decoder


class FMNISTDecoder(Decoder):
    """
    MNIST model decoder
    """

    def __init__(self, code_length: int, output_shape: Tuple[int, int, int]):
        """

        :param code_length:
        :param output_shape:
        """
        super(FMNISTDecoder, self).__init__(code_length=code_length,
                                            output_shape=output_shape)
        c, h, w = self.output_shape
        self.deepest_shape = (64, h // 4, w // 4)

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
