import pytest

import torch

from vintage_models.vision_transformers.vit.vit import ViTEncoder, ViTEmbedder, ViT


GPU_NOT_AVAILABLE = not torch.cuda.is_available()


@pytest.fixture
def image():
    return torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.float32)


@pytest.fixture
def embedded_batches():
    return torch.randint(0, 255, (1, 4, 64), dtype=torch.float32)


@pytest.fixture
def batches():
    return torch.randint(0, 255, (1, 4, 768), dtype=torch.float32)


class TestVitEncoder:
    vit_encoder = ViTEncoder(
        layer_count=2,
        head_count=2,
        embedding_len=64,
        mlp_hidden_size=16,
    )

    def test_simple(self, embedded_batches):
        output = self.vit_encoder(embedded_batches)
        assert output.shape == (1, 4, 64)

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, embedded_batches):
        self.vit_encoder.to("cuda")
        self.vit_encoder(embedded_batches.to("cuda"))


class TestVitEmbedder:
    vit_embedder = ViTEmbedder(
        patch_size=16,
        image_width=32,
        image_height=32,
        embedding_len=64,
    )

    def test_simple(self, batches):
        output = self.vit_embedder(batches)
        assert output.shape == (1, 5, 64)

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, batches):
        self.vit_embedder.to("cuda")
        self.vit_embedder(batches.to("cuda"))


class TestVit:
    vit = ViT(
        patch_size=16,
        image_width=32,
        image_height=32,
        embedding_len=64,
        mlp_hidden_size=16,
        head_count=2,
        layer_count=2,
        class_count=3,
    )

    def test_simple(self, image):
        output = self.vit(image)
        assert output.shape == (
            1,
            3,
        )
        assert torch.allclose(output.sum(), torch.tensor(1.0))

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, image):
        self.vit.to("cuda")
        self.vit(image.to("cuda"))
