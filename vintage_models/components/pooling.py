from functools import partial

import torch
from torch import Tensor, randn
from torch.nn import Module, Parameter
from torch.nn.functional import conv2d


class TrainableAddPool2D(Module):
    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        activation: Module,
        padding: str | int | tuple[int, int] = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.bias: None | Parameter = Parameter(randn(1))
        self.scale: None | Parameter = Parameter(randn(1))
        self.add_pool_2d = AddPool2D(
            kernel_size=kernel_size,
            in_channels=in_channels,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.add_pool_2d(x)

        if self.scale is not None:
            x *= self.scale

        if self.bias is not None:
            x += self.bias

        return self.activation(x)


class AddPool2D(Module):
    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        padding: str | int | tuple[int, int] = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = partial(
            conv2d,
            weight=torch.ones(1, in_channels, kernel_size, kernel_size),
            stride=kernel_size,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
