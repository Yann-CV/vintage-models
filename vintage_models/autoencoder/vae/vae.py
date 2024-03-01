from functools import partial

import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import Module, Linear, Tanh, Softplus
from torch.nn.functional import binary_cross_entropy, sigmoid

from vintage_models.components.multilayer_perceptron import LinearWithActivation


class VaeEncoder(Module):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        hidden_size: int,
        latent_size: int,
    ) -> None:
        super().__init__()
        in_size = image_width * image_height
        self.to_vector = partial(torch.reshape, shape=(-1, in_size))

        self.linear_with_tanh = LinearWithActivation(
            in_size=in_size, out_size=hidden_size, activation_layer=Tanh()
        )
        self.std_linear = LinearWithActivation(
            in_size=hidden_size, out_size=latent_size, activation_layer=Softplus()
        )
        self.mean_linear = Linear(hidden_size, latent_size)

        self._epsilon = torch.tensor(1e-6)

    def compute_mean_and_std(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.size(-3) != 1:
            raise ValueError(f"Expected the input to have 1 channel, got {x.size(-3)}")

        activated = self.linear_with_tanh(self.to_vector(x))

        mean = self.mean_linear(activated)
        std = self._epsilon + 10 ** self.std_linear(activated)

        return mean, std

    def forward(self, x: Tensor) -> Tensor:
        mean, std = self.compute_mean_and_std(x)
        return mean + std * torch.normal(
            mean=0, std=1, size=std.size(), device=std.device
        )


class VaeDecoder(Module):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        hidden_size: int,
        latent_size: int,
    ) -> None:
        super().__init__()
        self.linear_with_tanh = LinearWithActivation(
            in_size=latent_size, out_size=hidden_size, activation_layer=Tanh()
        )

        out_size = image_width * image_height
        self.std_linear = LinearWithActivation(
            in_size=hidden_size, out_size=out_size, activation_layer=Softplus()
        )
        self.mean_linear = Linear(hidden_size, out_size)

        self.to_image = partial(torch.reshape, shape=(-1, 1, image_width, image_height))
        self._epsilon = torch.tensor(1e-6)

    def forward(self, x: Tensor) -> Tensor:
        activated = self.linear_with_tanh(x)
        mean = self.mean_linear(activated)
        std = self._epsilon + 10 ** self.std_linear(activated)

        random_value = torch.normal(mean=0, std=1, size=std.size(), device=std.device)
        sampled = sigmoid(mean + std * random_value)

        return self.to_image(sampled)


class Vae(Module):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        hidden_size: int,
        latent_size: int,
        device: str | torch.device | int = "cuda",
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.image_width = image_width
        self.image_height = image_height
        self.hidden_size = hidden_size
        self.device = device
        self.encoder = VaeEncoder(
            image_width, image_height, hidden_size, latent_size
        ).to(device)
        self.decoder = VaeDecoder(
            image_width, image_height, hidden_size, latent_size
        ).to(device)

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def loss(self, x: Tensor) -> Normal:
        reconstructed = self.forward(x)

        vector_size = x.size(-1) * x.size(-2)
        reconstruction_loss = binary_cross_entropy(
            reconstructed.reshape(-1, vector_size),
            x.reshape(-1, vector_size),
            reduction="none",
        )

        mean, std = self.encoder.compute_mean_and_std(x)
        kl_div = 0.5 * (mean.pow(2) + std - torch.log(std) - 1)

        return reconstruction_loss.sum(dim=1).mean() + kl_div.sum(dim=1).mean()

    def generate(self, n: int) -> Tensor:
        return self.decoder(torch.randn(n, self.latent_size, device=self.device))

    def __str__(self) -> str:
        return (
            f"VAE_image_width_{self.image_width}_image_height_{self.image_height}"
            f"_hidden_size_{self.hidden_size}_latent_size_{self.latent_size}"
        )
