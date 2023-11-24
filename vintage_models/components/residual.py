from torch.nn import Module, ModuleList
from torch import Tensor


class ResidualWithSelfAttention(Module):
    """Residual module allowing to deal with self attention layers.

    The provided value is added to the result of the forward pass among all provided layers.

    Attributes:
        layers: The layers to apply.
    """

    def __init__(self, layers: list[Module]) -> None:
        super().__init__()
        self.layers = ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        init = x.clone()
        for layer in self.layers:
            x = layer(x, x, x) if getattr(layer, "qvk_module", False) else layer(x)
        return init + x
