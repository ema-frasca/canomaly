from functools import reduce
from operator import mul
from typing import Tuple
import abc
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    MNIST model encoder.
    """

    def __init__(self, input_shape: Tuple[int, int, int], code_length: int, variational: bool = False,
                 conditional: bool = False
                 ) -> None:
        """
        Class constructor:
        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        :param variational: whether to be variational or not
        :param conditional: whether to be conditional or not
        """
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.variational = variational
        self.conditional = conditional
        self.activation_fn = nn.LeakyReLU()

        # Convolutional network
        self._set_conv_block()

        self._set_conditional()

        self._set_fc_block()

    def _set_fc_block(self):
        # FC network
        if self.variational:
            # two layer, one for the mu and one for the logvar
            self.fc = nn.ModuleList([nn.Linear(in_features=reduce(mul, self.deepest_shape),
                                               out_features=self.code_length) for _ in range(2)])
            # todo: divide out_features by 2 to have the same number of parameters between vae and ae
        else:
            # only one layer
            self.fc = nn.ModuleList([nn.Linear(in_features=reduce(mul, self.deepest_shape),
                                    out_features=self.code_length)])

    @abc.abstractmethod
    def _set_conv_block(self):
        """
        set conv block and the encoder's deepest shape
        :return:
        """
        pass

    def _set_conditional(self):
        """
        add conditional layer
        :return:
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.
        :param x: the input batch of images.
        :return: the batch of latent vectors.
        """

        h = x
        h = self.conv(h)
        h = h.view(len(h), -1)
        o = torch.concat([fc(h) for fc in self.fc])

        return o
