import pytest
import torch

from vintage_models.adversarial.gan.gan import (
    GanDiscriminator,
    GanGenerator,
    Gan,
)


GPU_NOT_AVAILABLE = not torch.cuda.is_available()


@pytest.fixture
def input():
    return torch.randint(0, 256, (2, 1, 28, 28)).float() / 255


@pytest.fixture
def latent_input():
    return torch.randint(0, 256, (2, 2)).float() / 255


class TestGanGenerator:
    generator = GanGenerator(out_width=28, out_height=28, input_size=2, latent_size=100)

    def test_simple(self, latent_input):
        output = self.generator(latent_input)
        assert output.shape == (2, 1, 28, 28)

    def test_wrong_input_size(self, latent_input):
        with pytest.raises(RuntimeError):
            self.generator(torch.zeros(2, 1, 3))

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, latent_input):
        self.generator.to("cuda")
        self.generator(latent_input.to("cuda"))


class TestGanDiscriminator:
    discriminator = GanDiscriminator(28, 28, 100, 2)

    def test_simple(self, input):
        output = self.discriminator(input)
        assert output.shape == (2, 1)

    def test_wrong_input_size(self, input):
        with pytest.raises(RuntimeError):
            self.discriminator(torch.zeros(4, 1, 28, 29))

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.discriminator.to("cuda")
        self.discriminator(input.to("cuda"))


class TestGan:
    gan = Gan(
        image_width=28,
        image_height=28,
        generator_input_size=100,
        generator_latent_size=500,
        discriminator_hidden_size=5,
        discriminator_maxout_depth=2,
    )

    def test_simple(self, input):
        output = self.gan(input)
        assert output.shape == (2, 1)
        assert torch.all((output >= 0) & (output <= 1))

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.gan.to("cuda")
        self.gan(input.to("cuda"))
