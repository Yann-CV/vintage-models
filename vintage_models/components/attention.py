from torch.nn import Module, Linear
from torch import Tensor, matmul, softmax, cat


class ScaledDotProductAttention(Module):
    def __init__(self, dk: int) -> None:
        super().__init__()
        self.dk = dk
        self.qvk_module = True

    def forward(self, keys: Tensor, values: Tensor, queries: Tensor) -> Tensor:
        if keys.shape[1] != self.dk or queries.shape[1] != self.dk:
            raise ValueError(f"keys and queries must be equal to {self.dk}")

        if keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values should have the same row count")

        dot_product = matmul(queries, other=keys.transpose(0, 1))
        val_weights = softmax(dot_product / self.dk, dim=-1)
        return matmul(val_weights, other=values)


class HeadAttention(Module):
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

        self.heads = [HeadAttention(dk=self.dk, dv=self.dv) for _ in range(self.h)]
        self.linear = Linear(self.h * self.dv, self.dv)

        self.qvk_module = True

    def forward(self, keys: Tensor, values: Tensor, queries: Tensor) -> Tensor:
        outputs = [head(keys, values, queries) for head in self.heads]
        return self.linear(cat(outputs, dim=-1))
