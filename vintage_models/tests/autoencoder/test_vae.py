import pytest
import torch

from vintage_models.autoencoder.vae.vae import VaeEncoder, VaeDecoder, Vae


GPU_NOT_AVAILABLE = not torch.cuda.is_available()


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

    def test_wrong_input_size(self, input):
        with pytest.raises(RuntimeError):
            self.encoder(torch.zeros(4, 1, 28, 29))

    def test_wrong_channel_count(self, input):
        with pytest.raises(ValueError):
            self.encoder(torch.zeros(4, 3, 28, 28))

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.encoder.to("cuda")
        self.encoder(input.to("cuda"))


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

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, latent):
        self.decoder.to("cuda")
        self.decoder(latent.to("cuda"))


class TestVae:
    vae = Vae(28, 28, 100, 2)

    def test_simple(self, input):
        output = self.vae(input)
        assert output.shape == (2, 1, 28, 28)
        assert torch.all((output >= 0) & (output <= 1))

    def test_loss(self, input):
        loss = self.vae.loss(input)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_generate(self, input):
        generated = self.vae.generate(2)
        assert generated.shape == (2, 1, 28, 28)

    def test_wrong_input_size(self, input):
        with pytest.raises(RuntimeError):
            self.vae(torch.zeros(4, 1, 28, 28 - 1))

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.vae.to("cuda")
        self.vae(input.to("cuda"))
