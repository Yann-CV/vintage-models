from functools import partial

from torch import Tensor, reshape, device as torch_device, randn, randn_like, exp
from torch.nn import Module, Linear, Tanh, ReLU, Sigmoid
from torch.nn.functional import binary_cross_entropy

from vintage_models.components.multilayer_perceptron import LinearWithActivation


class VaeEncoder(Module):
    """Encoder for the vintage variational autoencoder.

    The input image is unrolled as a vector before going through the network. Its
    values are expected to be between 0 and 1.

    Attributes:
        image_width: Width of the input image.
        image_height: Height of the input image.
        hidden_size: Size of the hidden layer (after application of the linear and activation layer).
        latent_size: Size of the latent space.
        to_vector: Function to unroll the input image as a vector.
        linear_with_tanh: Linear layer followed by a tanh activation layer.
        log_var_linear: Linear layer allowing to compute the
        log of the variance for the latent space distribution.
        mean_linear: Linear layer to compute the mean for the latent space distribution.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        hidden_size: int,
        latent_size: int,
    ) -> None:
        """Initializes the VaeEncoder.

        Args:
            image_width: Width of the ouput image.
            image_height: Height of the ouput image.
            hidden_size: Size of the hidden layer (after application of the linear and activation layer).
            latent_size: Size of the latent layer.
        """
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        in_size = image_width * image_height
        self.to_vector = partial(reshape, shape=(-1, in_size))

        self.linear_with_tanh = LinearWithActivation(
            in_size=in_size, out_size=hidden_size, activation_layer=Tanh()
        )
        self.log_var_linear = Linear(hidden_size, latent_size)
        self.mean_linear = Linear(hidden_size, latent_size)

    def compute_mean_and_log_var(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.size(-3) != 1:
            raise ValueError(f"Expected the input to have 1 channel, got {x.size(-3)}")

        activated = self.linear_with_tanh(self.to_vector(x))

        mean = self.mean_linear(activated)
        log_var = self.log_var_linear(activated)

        return mean, log_var

    def forward(self, x: Tensor) -> Tensor:
        mean, log_var = self.compute_mean_and_log_var(x)
        std = exp(0.5 * log_var)
        return mean + std * randn_like(std)


class VaeDecoder(Module):
    """Decoder for the vintage variational autoencoder.

    The values of the generated images are between 0 and 1.

    Attributes:
        image_width: Width of the output image.
        image_height: Height of the output image.
        hidden_size: Size of the hidden layer.
        latent_size: Size of the latent layer.
        linear_with_relu: Linear layer followed by a relu activation layer (from latent to hidden).
        linear_with_sigmoid: Linear layer followed by a sigmoid activation layer (from hidden to image).
        to_image: Function to reshape the output vector as an image.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        hidden_size: int,
        latent_size: int,
    ) -> None:
        """Initializes the VaeDecoder.

        Args:
            image_width: Width of the ouput image.
            image_height: Height of the ouput image.
            hidden_size: Size of the hidden layer (after application of the linear and activation layer).
            latent_size: Size of the latent layer.
        """
        super().__init__()

        self.image_width = image_width
        self.image_height = image_height
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.linear_with_relu = LinearWithActivation(
            in_size=latent_size, out_size=hidden_size, activation_layer=ReLU()
        )
        out_size = image_width * image_height
        self.linear_with_sigmoid = LinearWithActivation(
            in_size=hidden_size, out_size=out_size, activation_layer=Sigmoid()
        )

        self.to_image = partial(reshape, shape=(-1, 1, image_width, image_height))

    def forward(self, x: Tensor) -> Tensor:
        sampled = self.linear_with_sigmoid(self.linear_with_relu(x))

        return self.to_image(sampled)


class Vae(Module):
    """Vintage implementation of the variational autoencoder.

    See the paper_review.md file for more information.

    In the encoder, the input image is unrolled as a vector before going through the network.
    The image generation is done from sampling a normal distribution and passing it through the decoder.

    The input image values are expected to be between 0 and 1. likewise, the generated image values
    will be between 0 and 1.

    Attributes:
        encoder: Encoder for the variational autoencoder. Tranforms the image toward the latent space.
        decoder: Decoder for the variational autoencoder. Tranforms the latent space toward the image space.
        device: Device to use for the model running.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        hidden_size: int,
        latent_size: int,
        device: str | torch_device | int = "cpu",
    ) -> None:
        """Initializes the Vae.

        Args:
            image_width: Width of the input and ouput images.
            image_height: Height of the input and ouput images.
            hidden_size: Size of the hidden layer (after application of the linear and activation layer).
            latent_size: Size of the latent layer.
            device: Device to use for the model running.
        """
        super().__init__()
        self.device = torch_device(device)

        self.latent_size = latent_size

        self.encoder = VaeEncoder(
            image_width, image_height, hidden_size, latent_size
        ).to(self.device)
        self.decoder = VaeDecoder(
            image_width, image_height, hidden_size, latent_size
        ).to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def loss(self, x: Tensor) -> Tensor:
        reconstructed = self.forward(x)

        vector_size = x.size(-1) * x.size(-2)
        reconstruction_loss = (
            binary_cross_entropy(
                reconstructed,
                x,
                reduction="none",
            )
            .reshape(-1, vector_size)
            .sum(dim=1)
        )

        mean, log_var = self.encoder.compute_mean_and_log_var(x)
        kl_div = -0.5 * (1 + log_var - log_var.exp() - mean.pow(2)).sum(dim=1)

        loss = kl_div.mean() + reconstruction_loss.mean()

        return loss

    def generate(self, n: int) -> Tensor:
        return self.decoder(randn(n, self.decoder.latent_size, device=self.device))

    def __str__(self) -> str:
        return (
            f"VAE_image_width_{self.encoder.image_width}_image_height_{self.encoder.image_height}"
            f"_hidden_size_{self.encoder.hidden_size}_latent_size_{self.encoder.latent_size}"
        )
