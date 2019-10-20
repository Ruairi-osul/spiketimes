import pytest
from spiketimes.alignment import align_to
import numpy as np


class TestAlignTo:
    def test_two_scaler_input(self):
        bad_input_1 = 1
        bad_input_2 = 2
        with pytest.raises(TypeError):
            align_to(to_be_aligned=bad_input_1, to_align_to=bad_input_2)

    def test_list_input(self):
        bad_input_1 = [3, 2, 1, 3, 2]
        bad_input_2 = [2, 3, 10, 2]
        good_input = np.array(bad_input_1)

        # one bad input
        with pytest.raises(TypeError):
            align_to(bad_input_1, good_input)
            align_to(good_input, bad_input_2)

        # one two bad inputs
        with pytest.raises(TypeError):
            align_to(bad_input_1, bad_input_2)

    def test_1d_array(self):
        # input arrays should be flat
        bad_input = np.array([3, 2, 1, 3, 2]).reshape(5, 1)
        good_input = np.array([32, 20, 10, 3, 2])

        with pytest.raises(ValueError):
            align_to(bad_input, good_input)
            align_to(good_input, bad_input)

    def test_events_before_first_aligning_event(self):
        a = np.arange(0, 10, 1)
        b = np.arange(3.5, 8.5, 1)
        good_output = np.array(
            [np.nan, np.nan, np.nan, np.nan, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5]
        )
        actual_output = align_to(a, b)
        np.testing.assert_allclose(actual_output, good_output)

    def test_normal_case(self):
        a = np.arange(0, 10, 2)
        b = np.arange(0.6, 8, 2)
        expected = np.array([np.nan, 1.4, 1.4, 1.4, 1.4])
        actual = align_to(a, b)
        np.testing.assert_allclose(actual, expected)

    def test_no_beyond(self):
        a = np.arange(0, 10, 2)
        b = np.arange(1.1, 3, 0.5)
        expected = np.array([np.nan, 0.4])
        actual = align_to(a, b, no_beyond=True)
        np.testing.assert_allclose(actual, expected)

