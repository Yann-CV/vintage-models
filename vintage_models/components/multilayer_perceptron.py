from torch.nn import Module, Linear, ReLU
from torch import Tensor


class TransformerMLP(Module):
    def __init__(self, io_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear_in = Linear(io_size, hidden_size)
        self.activation = ReLU()
        self.linear_out = Linear(hidden_size, io_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.linear_in(x))
        return self.linear_out(x)
