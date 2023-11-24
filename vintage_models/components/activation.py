from torch import Tensor
from torch.nn import Tanh


class ScaledTanh(Tanh):
    """Scaled tanh activation function.

    torch.nn.Tanh result is multiplied by a fixed scale factor.

    Args:
        scale: The scale factor to apply to the output of the tanh function.
    """

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) * self.scale
