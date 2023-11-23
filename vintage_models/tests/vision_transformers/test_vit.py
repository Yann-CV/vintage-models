import pytest

import torch

from vintage_models.vision_transformers.vit.vit import ViTEncoder, ViTEmbedder, ViT


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
    def test_simple(self, embedded_batches):
        encoder = ViTEncoder(
            layer_count=2,
            head_count=2,
            embedding_len=64,
            mlp_hidden_size=16,
        )
        output = encoder(embedded_batches)
        assert output.shape == (1, 4, 64)


class TestVitEmbedder:
    def test_simple(self, batches):
        embedder = ViTEmbedder(
            patch_size=16,
            image_width=32,
            image_height=32,
            embedding_len=64,
        )
        output = embedder(batches)
        assert output.shape == (1, 5, 64)


class TestVit:
    def test_simple(self, image):
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
        output = vit(image)
        assert output.shape == (
            1,
            3,
        )
        assert torch.allclose(output.sum(), torch.tensor(1.0))
