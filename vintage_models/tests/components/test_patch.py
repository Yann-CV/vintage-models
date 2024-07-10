import pytest

import torch

from vintage_models.components.patch import PatchConverter


GPU_NOT_AVAILABLE = not torch.cuda.is_available()


@pytest.fixture()
def input():
    return torch.arange(start=0, end=4 * 4 * 3, dtype=torch.float32).reshape(
        (1, 3, 4, 4)
    )


class TestPatchConverter:
    converter = PatchConverter(patch_size=2, image_width=4, image_height=4)

    def test_without_pad(self, input):
        output = self.converter(input)
        assert output.shape == (1, 4, 12)

    def test_with_pad(self, input):
        converter = PatchConverter(patch_size=5, image_width=4, image_height=4)
        output = converter(input)
        assert output.shape == (1, 1, 5 * 5 * 3)

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.converter.to("cuda")
        self.converter(input.to("cuda"))
