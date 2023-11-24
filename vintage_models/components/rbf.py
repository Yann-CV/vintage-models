import torch
from torch.nn import Linear, MSELoss

from torch.nn.functional import mse_loss


class EuclideanDistanceRBF(Linear):
    """Radial Basis Function layer computing euclidean distances.

    Each output is the euclidean distance between the input and the weight.

    Args:
        in_features: input size for each element in the batch.
        out_features: Size of output for each element in the bacth.
        Other: See `torch.nn.Linear`.
    """

    def __init__(
        self, in_features: int, out_features: int, device=None, dtype=None
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.mse = MSELoss(
            reduction="none",
        )

    def forward(self, x):
        loss = mse_loss(x, self.weight, reduction="none")
        return torch.sum(loss, dim=-1)
