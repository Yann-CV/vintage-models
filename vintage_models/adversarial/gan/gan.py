from collections import OrderedDict
from functools import partial

from torch import Tensor, device as torch_device, randn, reshape
from torch.nn import Module, Sequential, ReLU, Sigmoid, Dropout

from vintage_models.components.multilayer_perceptron import LinearWithActivation, MaxOut


class GanGenerator(Module):
    """Generator for the vintage generative adversarial network.

    Attributes:
        out_width: Width of the ouput image.
        out_height: Height of the ouput image.
        hidden_size: Size of the hidden layer (after application of the linear and activation layer).
        latent_size: Size of the latent space.
    """

    def __init__(
        self,
        out_width: int,
        out_height: int,
        hidden_size: int,
        latent_size: int,
    ) -> None:
        """Initializes the GAN generator.

        Args:
            out_width: Width of the ouput image.
            out_height: Height of the ouput image.
            hidden_size: Size of the hidden layer (after application of the linear and activation layer).
            latent_size: Size of the latent layer.
        """
        super().__init__()
        self.out_width = out_width
        self.out_height = out_height
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.model = Sequential(
            OrderedDict(
                [
                    (
                        "linear_with_relu_1",
                        LinearWithActivation(latent_size, hidden_size, ReLU()),
                    ),
                    (
                        "linear_with_relu_2",
                        LinearWithActivation(hidden_size, hidden_size, ReLU()),
                    ),
                    (
                        "linear_with_sigmoid",
                        LinearWithActivation(
                            hidden_size, self.out_width * self.out_height, Sigmoid()
                        ),
                    ),
                ]
            )
        )

        self.to_image = partial(reshape, shape=(-1, 1, self.out_width, self.out_height))

    def forward(self, x: Tensor) -> Tensor:
        generated = self.model(x)

        return generated.reshape(-1, 1, self.out_width, self.out_height)


class GanDiscriminator(Module):
    """Discriminator for the vintage variational autoencoder.

    Attributes:
        in_width: Width of the input image.
        in_height: Height of the input image.
        hidden_size: Size of the hidden layer.
        latent_size: Size of the latent layer.
    """

    def __init__(
        self,
        in_width: int,
        in_height: int,
        hidden_size: int,
        maxout_depth: int,
    ) -> None:
        """Initializes the GAN discriminator.

        Args:
            in_width: Width of the input image.
            in_height: Height of the input image.
            hidden_size: Size of the hidden layer (after application of the linear and activation layer).
            latent_size: Size of the latent layer.
        """
        super().__init__()

        self.in_width = in_width
        self.in_height = in_height
        self.hidden_size = hidden_size
        self.maxout_depth = maxout_depth

        in_size = in_width * in_height
        self.to_vector = partial(reshape, shape=(-1, in_size))

        self.model = Sequential(
            OrderedDict(
                [
                    ("maxout_hidden_1", MaxOut(in_size, hidden_size, maxout_depth)),
                    ("dropout_1", Dropout(0.5)),
                    ("maxout_hidden_2", MaxOut(hidden_size, hidden_size, maxout_depth)),
                    ("dropout_2", Dropout(0.5)),
                    (
                        "linear_with_sigmoid",
                        LinearWithActivation(hidden_size, 1, Sigmoid()),
                    ),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.to_vector(x)

        return self.model(x)


class Gan(Module):
    """Vintage implementation of the generative adversarial network.

    See the paper_review.md file for more information.

    Attributes:
        generator: Generator for the generative adversarial network. Tranform the noise in an image.
        discriminator: Discriminator for the generative adversarial network. Decide if the image
        is real or fake.
        device: Device to use for the model running.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        hidden_size: int,
        latent_size: int,
        maxout_depth: int,
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

        self.generator = GanGenerator(
            out_width=image_width,
            out_height=image_height,
            hidden_size=hidden_size,
            latent_size=latent_size,
        ).to(self.device)

        self.discriminator = GanDiscriminator(
            in_width=image_width,
            in_height=image_height,
            hidden_size=hidden_size,
            maxout_depth=maxout_depth,
        ).to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        return self.discriminator(x)

    def generate(self, n: int) -> Tensor:
        return self.generator(randn(n, self.generator.latent_size, device=self.device))

    def loss(self, x: Tensor) -> Tensor:
        pass

    def __str__(self) -> str:
        return (
            f"GAN_image_width_{self.generator.out_width}_image_height_{self.generator.out_height}"
            f"_hidden_size_{self.generator.hidden_size}_latent_size_{self.generator.latent_size}"
        )
