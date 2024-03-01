import pytest
import torch

from vintage_models.autoencoder.vae.vae import VaeEncoder, VaeDecoder, Vae


@pytest.fixture
def input():
    return torch.randint(0, 256, (2, 1, 28, 28)).float() / 255


@pytest.fixture
def latent():
    return torch.randn(2, 2)


class TestVaeEncoder:
    encoder = VaeEncoder(28, 28, 100, 2)

    def test_simple(self, input):
        output = self.encoder(input)
        assert output.shape == (2, 2)

    def test_distribution_from_sample(self, input):
        distribution = self.encoder.distribution_from_sample(input)
        assert distribution.sample().shape == (2, 2)

    def test_wrong_input_size(self, input):
        with pytest.raises(RuntimeError):
            self.encoder(torch.zeros(4, 1, 28, 29))

    def test_wrong_channel_count(self, input):
        with pytest.raises(ValueError):
            self.encoder(torch.zeros(4, 3, 28, 28))


class TestVaeDecoder:
    decoder = VaeDecoder(28, 28, 100, 2)

    def test_simple(self, latent):
        output = self.decoder(latent)
        assert output.shape == (2, 1, 28, 28)
        assert torch.all((output >= 0) & (output <= 1))

    def test_wrong_input_size(self, input):
        with pytest.raises(RuntimeError):
            self.decoder(torch.zeros(4, 1))

    def test_wrong_channel_count(self, input):
        with pytest.raises(RuntimeError):
            self.decoder(torch.zeros(4, 3, 28, 28))


class TestVae:
    vae = Vae(28, 28, 100, 2)

    def test_simple(self, input):
        output = self.vae(input)
        assert output.shape == (2, 1, 28, 28)
        assert torch.all(output != input)
        assert torch.all((output >= 0) & (output <= 1))

    def test_loss(self, input):
        loss = self.vae.loss(input)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_wrong_input_size(self, input):
        with pytest.raises(RuntimeError):
            self.vae(torch.zeros(4, 1, 28, 28 - 1))
