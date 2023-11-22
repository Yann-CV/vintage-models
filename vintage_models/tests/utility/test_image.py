from vintage_models.utility.transform import padding_values_to_be_multiple


class TestPaddingValuesToBeMultiple:
    def test_with_smaller_real(self):
        real = 10
        multiple_of = 15
        start, end = padding_values_to_be_multiple(real, multiple_of)
        assert (real + start + end) == multiple_of

    def test_with_equal(self):
        real = 15
        multiple_of = 15
        start, end = padding_values_to_be_multiple(real, multiple_of)
        assert start == end == 0

    def test_with_already_multiple(self):
        real = 30
        multiple_of = 15
        start, end = padding_values_to_be_multiple(real, multiple_of)
        assert start == end == 0

    def test_with_bigger_real(self):
        real = 25
        multiple_of = 15
        start, end = padding_values_to_be_multiple(real, multiple_of)
        assert ((real + start + end) % multiple_of) == 0
