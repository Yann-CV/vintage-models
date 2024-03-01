from collections import OrderedDict

from torch.nn import Module, Linear, GELU, Sequential
from torch import Tensor


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
