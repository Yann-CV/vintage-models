from collections import OrderedDict

from torch.nn import Module, Linear, GELU, Sequential
from torch import Tensor


class TwoLayerGeluMLP(Module):
    def __init__(self, in_size: int, out_size: int, hidden_size: int) -> None:
        super().__init__()
        self.model = Sequential(
            OrderedDict(
                [
                    ("linear_in", Linear(in_size, hidden_size)),
                    ("activation", GELU()),
                    ("conv2", Linear(hidden_size, out_size)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class TwoLayerMLP(Module):
    def __init__(self, in_size: int, out_size: int, hidden_size: int) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size

        self.model = Sequential(
            OrderedDict(
                [
                    ("linear_in", Linear(in_size, hidden_size)),
                    ("conv2", Linear(hidden_size, out_size)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
