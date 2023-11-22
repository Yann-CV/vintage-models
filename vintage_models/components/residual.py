from torch.nn import Module
from torch import Tensor


class ResidualWithSelfAttention(Module):
    def __init__(self, layers: list[Module]) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        init = x.clone()
        for layer in self.layers:
            x = layer(x, x, x) if getattr(layer, "qvk_module", False) else layer(x)
        return init + x
