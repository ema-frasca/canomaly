from typing import Tuple

from torch import nn

from backbones.blocks_2d import DownsampleBlock, ResidualBlock
from backbones.utils.encoder import Encoder


class CIFAR10Encoder(Encoder):
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
        super(CIFAR10Encoder, self).__init__(input_shape=input_shape, code_length=code_length, variational=variational,
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