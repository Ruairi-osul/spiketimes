import pytest
from spiketimes.alignment import align_to, negative_align
from spiketimes.alignment.binning import (
    binned_spiketrain,
    binned_spiketrain_bins_provided,
    bin_to_bool,
    which_bin,
    spike_count_around_event,
)
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
        expected = np.array([np.nan, 0.4, np.nan, np.nan, np.nan])
        actual = align_to(a, b, no_beyond=True)
        np.testing.assert_allclose(actual, expected)


class TestNegativeAlign:
    def test_two_scaler_input(self):
        bad_input_1 = 1
        bad_input_2 = 2
        with pytest.raises(TypeError):
            negative_align(to_be_aligned=bad_input_1, to_align_to=bad_input_2)

    def test_list_input(self):
        bad_input_1 = [3, 2, 1, 3, 2]
        bad_input_2 = [2, 3, 10, 2]
        good_input = np.array(bad_input_1)

        # one bad input
        with pytest.raises(TypeError):
            negative_align(bad_input_1, good_input)
            negative_align(good_input, bad_input_2)

        # one two bad inputs
        with pytest.raises(TypeError):
            negative_align(bad_input_1, bad_input_2)

    def test_1d_array(self):
        # input arrays should be flat
        bad_input = np.array([3, 2, 1, 3, 2]).reshape(5, 1)
        good_input = np.array([32, 20, 10, 3, 2])

        with pytest.raises(ValueError):
            negative_align(bad_input, good_input)
            negative_align(good_input, bad_input)

    def test_events_after_first_aligning_event(self):
        a = np.arange(7, 10, 1)
        b = np.arange(3.5, 10.5, 1)
        good_output = np.array([-0.5, -0.5, np.nan])
        actual_output = negative_align(a, b)
        np.testing.assert_allclose(actual_output, good_output)

    def test_normal_case(self):
        a = np.arange(0, 10, 2)
        b = np.arange(0.6, 8, 2)
        expected = np.array([-0.6, -0.6, -0.6, np.nan, np.nan])
        actual = negative_align(a, b)
        np.testing.assert_allclose(actual, expected)

    def test_no_beyond(self):
        a = np.arange(0, 10, 2)
        b = np.arange(1.1, 3, 0.5)
        expected = np.array([np.nan, -0.1, np.nan, np.nan, np.nan])
        actual = negative_align(a, b, no_before=True)
        np.testing.assert_allclose(actual, expected)


class AlignAround:
    def test_normal(self):
        pass


class TestBinnedSpikeTrain:
    def test_normal(self):
        FS = 2
        T_START = 0
        spiketrain = np.array([0.5, 0.6, 1.1, 1.2, 2.1, 3])

        expected_bins = np.array([0.5, 1, 1.5, 2, 2.5, 3])
        expected_values = np.array([0, 2, 2, 0, 1, 1])

        actual_bins, actual_values = binned_spiketrain(
            spiketrain=spiketrain, fs=FS, t_start=T_START
        )
        np.testing.assert_allclose(actual_bins, expected_bins)
        np.testing.assert_allclose(actual_values, expected_values)


class TestBinsProvided:
    def test_normal(self):
        bins = np.arange(0, 2.6, 0.5)
        spiketimes = np.array([0.1, 0.2, 0.2, 0.8, 1.1, 1.5, 1.9, 2.2])

        expected = np.array([3, 1, 1, 2, 1])
        actual = binned_spiketrain_bins_provided(spiketimes, bins)
        np.testing.assert_allclose(actual, expected)


class TestBinToBool:
    def test_normal(self):
        binned_arr = np.array([0, 10, 20, 0, 3, 1, 0, 0, 0, 2])
        expected = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 1])
        actual = bin_to_bool(binned_arr)
        np.testing.assert_allclose(actual, expected)


class TestWhichBin:
    def test_normal(self):
        spiketimes = np.array([0.1, 0.2, 0.3, 0.6, 1.1])
        bins = np.arange(0, 2, 0.5)

        expected_idx = np.array([0, 0, 0, 1, 2])
        expected_vals = np.array([0.5, 0.5, 0.5, 1, 1.5])

        actual_idx, actual_values = which_bin(spiketimes, bin_edges=bins)

        np.testing.assert_allclose(actual_idx, expected_idx)
        np.testing.assert_allclose(actual_values, expected_vals)


class TestSpikeCountAroundEvent:
    def test_normal(self):
        BINSIZE = 0.2
        events = np.arange(0, 4.1, 0.5)
        spiketrain = np.array([0.1, 0.3, 0.51, 0.56, 0.7, 0.72, 1.1, 1.9, 2, 2.5, 3.6])

        expected = np.array([1, 2, 1, 0, 1, 1, 0, 1, 0])
        actual = spike_count_around_event(spiketrain, events, binsize=BINSIZE)

        np.testing.assert_allclose(actual, expected)
