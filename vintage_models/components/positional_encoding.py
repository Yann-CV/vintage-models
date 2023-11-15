from torch.nn import Module, ParameterList, Parameter
from torch import Tensor, sin, cos, tensor, stack, randn


class PositionalEncoding1D(Module):
    def __init__(self, sequence_len: int, n: int, embedding_len: int) -> None:
        super().__init__()
        self.sequence_len = sequence_len
        self.n = n
        if embedding_len % 2 != 0:
            raise ValueError(
                f"The length {embedding_len} of an embedding must be even."
            )
        self.embedding_len = embedding_len
        self.positional_embeddings = self._compute_positional_embeddings()

    def _compute_positional_embeddings(self):
        sin_cos_len = self.embedding_len // 2
        k = tensor([[idx] * sin_cos_len for idx in range(0, self.sequence_len)])
        i = tensor([range(0, sin_cos_len)] * self.sequence_len)
        sin_results = self._sin(k, i)
        cos_results = self._cos(k, i)

        result = [
            sin_results[:, embedding_idx // 2]
            if embedding_idx % 2 == 0
            else cos_results[:, embedding_idx // 2]
            for embedding_idx in range(0, self.embedding_len)
        ]

        return stack(result, dim=1)

    def _sin(self, k: Tensor, i: Tensor) -> Tensor:
        return sin(k / self.n ** (2 * i / self.embedding_len))

    def _cos(self, k: Tensor, i: Tensor) -> Tensor:
        return cos(k / self.n ** ((2 * i + 1) / self.embedding_len))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.positional_embeddings


class LearnablePositionalEncoding1D(Module):
    def __init__(self, sequence_len: int, embedding_len: int) -> None:
        super().__init__()
        self.sequence_len = sequence_len
        self.embedding_len = embedding_len
        self.positional_embeddings = ParameterList(
            [Parameter(randn(1, embedding_len)) for _ in range(0, self.sequence_len)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for idx, positional_embedding in enumerate(self.positional_embeddings):
            x[idx, :] = x[idx, :] + positional_embedding

        return x
