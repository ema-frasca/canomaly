from typing import Tuple
import abc
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    MNIST model decoder.
    """

    def __init__(self, code_length: int, output_shape: Tuple[int, int, int]) -> None:
        """
        Class constructor.
        :param code_length: the dimensionality of latent vectors.
        :param output_shape: the shape of MNIST samples.
        """
        super(Decoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = None
        self.output_shape = output_shape
        self.activation_fn = nn.LeakyReLU()

    @abc.abstractmethod
    def _set_fc_block(self):
        """
        set fully connected block
        :return:
        """
        pass

    @abc.abstractmethod
    def _set_conv_block(self):
        """
        set conv block
        :return:
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.
        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x
        h = self.fc(h)
        h = h.view(len(h), *self.deepest_shape)
        h = self.conv(h)
        o = h

        return o
