import pytest
import torch

from vintage_models.adversarial.gan.gan import (
    GanDiscriminator,
    GanGenerator,
    Gan,
    GanLosses,
)


@pytest.fixture
def input():
    return torch.randint(0, 256, (2, 1, 28, 28)).float() / 255


@pytest.fixture
def latent_input():
    return torch.randint(0, 256, (2, 1, 2)).float() / 255


class TestGanGenerator:
    generator = GanGenerator(28, 28, 100, 2)

    def test_simple(self, latent_input):
        output = self.generator(latent_input)
        assert output.shape == (2, 1, 28, 28)

    def test_wrong_input_size(self, latent_input):
        with pytest.raises(RuntimeError):
            self.generator(torch.zeros(2, 1, 3))


class TestGanDiscriminator:
    discriminator = GanDiscriminator(28, 28, 100, 2)

    def test_simple(self, input):
        output = self.discriminator(input)
        assert output.shape == (2, 1)

    def test_wrong_input_size(self, input):
        with pytest.raises(RuntimeError):
            self.discriminator(torch.zeros(4, 1, 28, 29))


class TestGan:
    gan = Gan(28, 28, 100, 2, 5)

    def test_simple(self, input):
        output = self.gan(input)
        assert output.shape == (2, 1)
        assert torch.all((output >= 0) & (output <= 1))

    def test_generate(self, input):
        generated = self.gan.generate(2)
        assert generated.shape == (2, 1, 28, 28)

    def test_loss(self, input):
        losses = self.gan.loss(input)
        generator_loss = losses.generator_loss.item()
        discriminator_loss = losses.discriminator_loss.item()
        assert isinstance(losses, GanLosses)
        assert isinstance(generator_loss, float)
        assert isinstance(discriminator_loss, float)
