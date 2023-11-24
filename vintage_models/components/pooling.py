from torch import Tensor, randn
from torch.nn import Module, Parameter, ParameterList, AvgPool2d
from torch.nn.common_types import _size_2_t


class AddPool2D(AvgPool2d):
    """Add the values in the pooling kernel together.

    Attributes: See torch.nn.AvgPool2d. Here, the divisor_override is always 1
    (obtain the sum and not the average).
    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: None | _size_2_t = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=1,
        )


class TrainableAddPool2D(Module):
    """Add pooling with trainable scale, bias and activation.

    Attributes:
        in_channels: Number of input channels for each element of the batch.
        activation: Activation function to apply after the scaled and biased pooling.
        Other: see torch.nn.AvgPool2d. Here, the divisor_override is always 1
        (obtain the sum and not the average).
    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        in_channels: int,
        activation: Module,
        stride: None | _size_2_t = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.bias: ParameterList = ParameterList(
            [Parameter(randn(1)) for _ in range(in_channels)]
        )
        self.scale: ParameterList = ParameterList(
            [Parameter(randn(1)) for _ in range(in_channels)]
        )
        self.add_pool_2d = AddPool2D(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.add_pool_2d(x)
        for idx, s in enumerate(self.scale):
            x[:, idx, :, :] *= s
            x[:, idx, :, :] *= self.bias[idx]

        return self.activation(x)
