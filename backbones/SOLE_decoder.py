from functools import reduce
from typing import Tuple

from torch import nn, mul

from backbones.blocks_1d import UpsampleBlock
from backbones.utils.decoder import Decoder


class SOLEDecoder(Decoder):
    """
    MNIST model decoder
    """

    def __init__(self, code_length: int, output_shape: int):
        """

        :param code_length:
        :param output_shape:
        """
        self.deepest_shape = (output_shape // 8, )
        super(SOLEDecoder, self).__init__(code_length=code_length,
                                          output_shape=output_shape)
        self.output_shape = output_shape

    def _set_conv_block(self):
        # Convolutional network
        self.conv = UpsampleBlock(channel_in=self.deepest_shape[0], channel_out=self.output_shape,
                                  activation_fn=self.activation_fn)

    def _set_fc_block(self):
        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.code_length, out_features=reduce(mul, self.deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, self.deepest_shape)),
            self.activation_fn
            # nn.Linear(in_features=9, out_features=reduce(mul, self.deepest_shape)),
            # nn.BatchNorm1d(num_features=reduce(mul, self.deepest_shape)),
            # self.activation_fn
        )
