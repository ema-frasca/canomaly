from typing import Tuple

from torch import nn

from backbones.blocks_1d import DownsampleBlock
from backbones.utils.encoder import Encoder


class SOLEEncoder(Encoder):
    """
    MNIST model encoder.
    """

    def __init__(self, input_shape: int, code_length: int, variational: bool = False,
                 conditional: bool = False) -> None:
        """
        Class constructor:
        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        """
        self.deepest_shape = (1, 1, input_shape // 8)

        super(SOLEEncoder, self).__init__(input_shape=(1, 1, input_shape), code_length=code_length,
                                          variational=variational,conditional=conditional)

    def _set_conv_block(self):
        _, _, w = self.input_shape

        self.conv = DownsampleBlock(channel_in=w, channel_out=self.deepest_shape[-1], activation_fn=self.activation_fn)

