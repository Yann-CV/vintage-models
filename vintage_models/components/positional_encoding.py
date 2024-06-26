from torch.nn import Module, ParameterList, Parameter
from torch import Tensor, sin, cos, tensor, stack, randn, cat


class PositionalEncoding1D(Module):
    """Positional encoding for 1D sequences hardcoded from sinus and cosinus.

    Note: This module cannot be use as it to process minibatches.

    Attributes:
        sequence_len: The length of the sequence.
        n: The number to modulate (manage the period) the cos and sin functions.
        embedding_len: The length of the embedding.
        positional_embeddings: The positional embeddings to add to the input.
    """

    def __init__(self, sequence_len: int, n: int, embedding_len: int) -> None:
        super().__init__()
        self.sequence_len = sequence_len
        self.n = n
        if embedding_len % 2 != 0:
            raise ValueError(
                f"The length {embedding_len} of an embedding must be even."
            )
        self.embedding_len = embedding_len
        self.register_buffer(
            "positional_embeddings",
            self._compute_positional_embeddings(),
            persistent=True,
        )

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
    """Learnable positional encoding for 1D sequences.

    All positional embedding are initialized with random values. They can then be learned during
    the training process.

    Attributes:
        sequence_len: The length of the sequence.
        embedding_len: The length of the embedding.
        positional_embeddings: The positional embeddings to add to the input.
    """

    def __init__(self, sequence_len: int, embedding_len: int) -> None:
        super().__init__()
        self.sequence_len = sequence_len
        self.embedding_len = embedding_len
        self.positional_embeddings = ParameterList(
            [Parameter(randn(1, embedding_len)) for _ in range(0, self.sequence_len)]
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-2] != self.sequence_len:
            raise ValueError(
                f"The sequence length {x.shape[-2]} of the input does not match "
                f"the sequence length {self.sequence_len} of the positional embeddings."
            )

        with_positions = [
            cat(
                [
                    single[idx, :] + positional_embedding
                    for idx, positional_embedding in enumerate(
                        self.positional_embeddings
                    )
                ],
                dim=0,
            )
            for single in x
        ]

        return stack(with_positions, dim=0)
