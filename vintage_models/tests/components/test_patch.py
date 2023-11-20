import pytest

import torch
import numpy as np

from vintage_models.components.patch import PatchConverter


@pytest.fixture()
def input():
    matrix = np.array([v for v in range(4 * 4 * 3)], dtype=np.uint8).reshape(
        (1, 3, 4, 4)
    )
    return torch.tensor(matrix, dtype=torch.float32)


class TestPatchConverter:
    def test_without_pad(self, input):
        converter = PatchConverter(patch_size=2, image_width=4, image_height=4)
        output = converter(input)
        assert output.shape == (1, 4, 12)

    def test_with_pad(self, input):
        converter = PatchConverter(patch_size=5, image_width=4, image_height=4)
        output = converter(input)
        assert output.shape == (1, 1, 5 * 5 * 3)
