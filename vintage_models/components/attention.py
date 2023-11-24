from torch.nn import Module, Linear, ModuleList
from torch import Tensor, matmul, softmax, cat


class ScaledDotProductAttention(Module):
    """Scaled dot product attention.

    The queries and keys are multiplied together, then scaled and finally softmaxed to make
    the weighted sum over values.

    Attributes:
        dk: The length of the keys and queries.
        qvk_module: is True because this module is requesting queries, keys and values
        in the forward method.
    """

    def __init__(self, dk: int) -> None:
        super().__init__()
        self.dk = dk
        self.qvk_module = True

    def forward(self, keys: Tensor, values: Tensor, queries: Tensor) -> Tensor:
        if keys.shape[-1] != self.dk or queries.shape[-1] != self.dk:
            raise ValueError(f"keys and queries length must be equal to {self.dk}")

        if keys.shape[-2] != values.shape[-2]:
            raise ValueError("keys and values should have the same row count")

        dot_product = matmul(queries, other=keys.transpose(-2, -1))
        val_weights = softmax(dot_product / self.dk, dim=-1)
        return matmul(val_weights, other=values)


class HeadAttention(Module):
    """Head attention linearizing queries, keys and values before applying scaled dot product attention.

    Attributes:
        dk: The length of the keys and queries.
        dv: The length of the values.
        qvk_module: is True because this module is requesting queries, keys and values
        in the forward method.
        k_linear: Linear layer for linearizing keys.
        q_linear: Linear layer for linearizing queries.
        v_linear: Linear layer for linearizing values.
        attention: Scaled dot product attention module.
    """

    def __init__(self, dk: int, dv: int) -> None:
        super().__init__()
        self.dk = dk
        self.dv = dv

        self.k_linear = Linear(self.dk, self.dk)
        self.q_linear = Linear(self.dk, self.dk)
        self.v_linear = Linear(self.dv, self.dv)

        self.attention = ScaledDotProductAttention(dk=self.dk)
        self.qvk_module = True

    def forward(self, keys: Tensor, values: Tensor, queries: Tensor) -> Tensor:
        v_linearized = self.v_linear(values)
        k_linearized = self.k_linear(keys)
        q_linearized = self.q_linear(queries)

        return self.attention(k_linearized, v_linearized, q_linearized)


class MultiHeadAttention(Module):
    def __init__(self, dk: int, dv: int, h: int) -> None:
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.h = h

        self.heads = ModuleList(
            [HeadAttention(dk=self.dk, dv=self.dv) for _ in range(self.h)]
        )
        self.linear = Linear(self.h * self.dv, self.dv)

        self.qvk_module = True

    def forward(self, keys: Tensor, values: Tensor, queries: Tensor) -> Tensor:
        outputs = [head(keys, values, queries) for head in self.heads]
        return self.linear(cat(outputs, dim=-1))
