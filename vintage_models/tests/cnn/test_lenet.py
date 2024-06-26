import pytest
import torch

from vintage_models.cnn.lenet.lenet import LeNet5


GPU_NOT_AVAILABLE = not torch.cuda.is_available()


@pytest.fixture
def image():
    return torch.randint(0, 255, (2, 3, 28, 28), dtype=torch.float32)


class TestLeNet5:
    lenet5 = LeNet5(
        image_width=28,
        image_height=28,
        class_count=10,
    )

    def test_simple(self, image):
        output = self.lenet5(image)
        assert output.shape == (
            2,
            10,
        )
        assert torch.allclose(output[0, :].sum(), torch.tensor(1.0))

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, image):
        self.lenet5.to("cuda")
        self.lenet5(image.to("cuda"))
