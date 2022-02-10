from typing import Tuple

from torch import nn

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

        super(FMNISTEncoder, self).__init__(input_shape=input_shape, code_length=code_length, variational=variational,
                                           conditional=conditional)

    def _set_conv_block(self):
        c, h, w = self.input_shape

        self.conv = nn.Sequential(
            DownsampleBlock(channel_in=c, channel_out=32, activation_fn=self.activation_fn),
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=self.activation_fn),
        )
