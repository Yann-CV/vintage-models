import pytest
import torch

from vintage_models.components.image import ImageTargeter, MaybeToColor


@pytest.fixture()
def input():
    return torch.arange(start=0, end=4 * 4, dtype=torch.float32).reshape((1, 1, 4, 4))


class TestImageTargeter:
    def test_with_padding(self, input):
        targeter = ImageTargeter(width_in=4, height_in=4, width_out=6, height_out=6)
        output = targeter(input)
        assert output.shape == (1, 3, 6, 6)

    def test_with_resize(self, input):
        targeter = ImageTargeter(width_in=4, height_in=4, width_out=3, height_out=3)
        output = targeter(input)
        assert output.shape == (1, 3, 3, 3)

    def test_with_nothing(self, input):
        targeter = ImageTargeter(width_in=4, height_in=4, width_out=4, height_out=4)
        output = targeter(input)
        assert output.shape == (1, 3, 4, 4)

    def test_without_color(self, input):
        targeter = ImageTargeter(
            width_in=4, height_in=4, width_out=4, height_out=4, color=False
        )
        output = targeter(input)
        assert output.shape == (1, 1, 4, 4)

    def test_with_wrong_size(self):
        targeter = ImageTargeter(width_in=4, height_in=4, width_out=4, height_out=4)
        with pytest.raises(ValueError):
            targeter(torch.zeros(1, 3, 5, 5))


class TestMaybeToColor:
    def test_with_batch(self, input):
        assert MaybeToColor()(input).shape == (1, 3, 4, 4)

    def test_with_monochrome(self):
        image = torch.zeros(1, 5, 5)
        assert MaybeToColor()(image).shape == (3, 5, 5)

    def test_with_color(self):
        image = torch.zeros(3, 5, 5)
        assert MaybeToColor()(image).shape == (3, 5, 5)
