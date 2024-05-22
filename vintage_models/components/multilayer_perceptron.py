from collections import OrderedDict

from torch.nn import Module, Linear, GELU, Sequential
from torch import Tensor, stack, device as torch_device


class TwoLayerMLP(Module):
    """Two layer multilayer perceptron.

    Attributes:
        in_size: The size of the input.
        out_size: The size of the output layer.
        hidden_size: The size of the hidden layer.
        model: the model allowing to perform the forward pass.
    """

    def __init__(self, in_size: int, out_size: int, hidden_size: int) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size

        self.model = Sequential(
            OrderedDict(
                [
                    ("linear_in", Linear(in_size, hidden_size)),
                    ("linear_out", Linear(hidden_size, out_size)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class LinearWithActivation(Module):
    def __init__(self, in_size: int, out_size: int, activation_layer: Module) -> None:
        super().__init__()
        self.linear = Linear(in_size, out_size)
        self.activation = activation_layer

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.linear(x))


class MaxOut(Module):
    """Maxout layer

    The maxout layer is taking the maximum value of the hidden linear layers applied
    to the input.

    See the `Maxout networks` paper for more details.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        maxout_depth: int,
        device: str | torch_device | int = "cpu",
    ) -> None:
        super().__init__()
        self.linears = [
            Linear(in_features=in_features, out_features=out_features, device=device)
            for _ in range(maxout_depth)
        ]

    def forward(self, x: Tensor) -> Tensor:
        linear_outputs = stack([linear(x) for linear in self.linears], dim=2)
        return linear_outputs.max(dim=2).values


class TwoLayerGeluMLP(Module):
    """Two layer multilayer perceptron with GELU activation function.

    Attributes:
        model: the model allowing to perform the forward pass.
    """

    def __init__(self, in_size: int, out_size: int, hidden_size: int) -> None:
        """
        Args:
            in_size: The size of the input.
            out_size: The size of the output layer.
            hidden_size: The size of the hidden layer.
            model: the model allowing to perform the forward pass.
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.model = Sequential(
            OrderedDict(
                [
                    ("linear_in", Linear(in_size, hidden_size)),
                    ("activation", GELU()),
                    ("linear_out", Linear(hidden_size, out_size)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
