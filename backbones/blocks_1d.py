from typing import List
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Module


def residual_op(x: torch.Tensor, functions: List[Module], bns: Optional[List[Module]],
                activation_fn: Module) -> torch.Tensor:
    """
    Implements a global residual operation.
    :param x: the input tensor.
    :param functions: a list of functions (nn.Modules).
    :param bns: a list of optional batch-norm layers.
    :param activation_fn: the activation to be applied.
    :return: the output of the residual operation.
    """
    f1, f2, f3 = functions
    bn1, bn2, bn3 = bns

    assert len(functions) == len(bns) == 3
    assert f1 is not None and f2 is not None
    assert not (f3 is None and bn3 is not None)

    # A-branch
    ha = x
    ha = f1(ha)
    if bn1 is not None:
        ha = bn1(ha)
    ha = activation_fn(ha)

    ha = f2(ha)
    if bn2 is not None:
        ha = bn2(ha)

    # B-branch
    hb = x
    if f3 is not None:
        hb = f3(hb)
    if bn3 is not None:
        hb = bn3(hb)

    # Residual connection
    out = ha + hb
    return activation_fn(out)


class BaseBlock(Module):
    """ Base class for all blocks. """

    def __init__(self, channel_in: int, channel_out: int, activation_fn: Module, use_bn: bool = True,
                 use_bias: bool = False):
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(BaseBlock, self).__init__()

        assert not (use_bn and use_bias), 'Using bias=True with batch_normalization is forbidden.'

        self._channel_in = channel_in
        self._channel_out = channel_out
        self._activation_fn = activation_fn
        self._use_bn = use_bn
        self._bias = use_bias

    def get_bn(self, channel=None) -> Optional[Module]:
        """
        Returns batch norm layers, if needed.
        :return: batch norm layers or None
        """
        return nn.BatchNorm1d(num_features=self._channel_out if channel is None else channel) if self._use_bn else None

    def forward(self, x: torch.Tensor):
        """
        Abstract forward function. Not implemented.
        """
        raise NotImplementedError


class DownsampleBlock(BaseBlock):
    """ Implements a Downsampling block for images (Fig. 1ii). """

    def __init__(self, channel_in: int, channel_out: int, activation_fn: Module, use_bn: bool = True,
                 use_bias: bool = False):
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(DownsampleBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        # Convolutions
        linear1a = nn.Linear(in_features=channel_in, out_features=channel_in//2, bias=use_bias)
        linear1b = nn.Linear(in_features=channel_in//2, out_features=channel_in//4, bias=use_bias)
        linear1c = nn.Linear(in_features=channel_in//4, out_features=channel_out, bias=use_bias)

        # Batch Normalization layers
        bn1a = self.get_bn(channel_in//2)
        bn1b = self.get_bn(channel_in//4)
        bn1c = self.get_bn()

        self.ds_net = nn.Sequential(
            linear1a,
            bn1a,
            activation_fn,
            linear1b,
            bn1b,
            activation_fn,
            linear1c,
            bn1c,
            activation_fn
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return self.ds_net.forward(x)


class UpsampleBlock(BaseBlock):
    """ Implements a Upsampling block for images (Fig. 1ii). """

    def __init__(self, channel_in: int, channel_out: int, activation_fn: Module, use_bn: bool = True,
                 use_bias: bool = False):
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(UpsampleBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        linear1a = nn.Linear(in_features=channel_in, out_features=channel_in * 2, bias=use_bias)
        linear1b = nn.Linear(in_features=channel_in * 2, out_features=channel_in * 4, bias=use_bias)
        linear1c = nn.Linear(in_features=channel_in * 4, out_features=channel_out, bias=use_bias)

        # Batch Normalization layers
        bn1a = self.get_bn(channel_in * 2)
        bn1b = self.get_bn(channel_in * 4)
        bn1c = self.get_bn(channel_out)

        self.ds_net = nn.Sequential(
            linear1a,
            bn1a,
            activation_fn,
            linear1b,
            bn1b,
            activation_fn,
            linear1c,
            bn1c,
            activation_fn
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return self.ds_net(x)


# class ResidualBlock(BaseBlock):
#     """ Implements a Residual block for images (Fig. 1ii). """
#
#     def __init__(self, channel_in: int, channel_out: int, activation_fn: Module, use_bn: bool = True,
#                  use_bias: bool = False) -> None:
#         """
#         Class constructor.
#         :param channel_in: number of input channels.
#         :param channel_out: number of output channels.
#         :param activation_fn: activation to be employed.
#         :param use_bn: whether or not to use batch-norm.
#         :param use_bias: whether or not to use bias.
#         """
#         super(ResidualBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)
#
#         # Convolutions
#         self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
#                                padding=1, stride=1, bias=use_bias)
#         self.conv2 = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
#                                padding=1, stride=1, bias=use_bias)
#
#         # Batch Normalization layers
#         self.bn1 = self.get_bn()
#         self.bn2 = self.get_bn()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward propagation.
#         :param x: the input tensor
#         :return: the output tensor
#         """
#         return residual_op(
#             x,
#             functions=[self.conv1, self.conv2, None],
#             bns=[self.bn1, self.bn2, None],
#             activation_fn=self._activation_fn
#         )
