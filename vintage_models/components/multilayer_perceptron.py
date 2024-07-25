from collections import OrderedDict

from torch.nn import Module, Linear, GELU, Sequential, ModuleList, BatchNorm1d
from torch import Tensor, stack, device as torch_device, max as torch_max


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
    """Linear layer followed by a potential batch normalization and an activation layer.

    Attributes:
        linear: The linear layer.
        normalize: The batch normalization layer. None if no normalization is needed.
        activation: The activation layer.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        activation_layer: Module,
        normalize: bool = False,
    ) -> None:
        """Initializes the LinearWithActivation.

        Args:
            in_size: The size of the input.
            out_size: The size of the output layer.
            activation_layer: The activation layer.
            normalize: Whether to normalize the output of the linear layer
        """
        super().__init__()
        self.linear = Linear(in_size, out_size)
        self.normalize = BatchNorm1d(out_size) if normalize else None
        self.activation = activation_layer

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.normalize is not None:
            x = self.normalize(x)
        return self.activation(x)


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

        self.linears = ModuleList(
            [
                Linear(
                    in_features=in_features, out_features=out_features, device=device
                )
                for _ in range(maxout_depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        linear_outputs = stack([linear(x) for linear in self.linears], dim=2)
        return torch_max(linear_outputs, dim=2)[0]


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
