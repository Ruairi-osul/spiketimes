import pytest
import numpy as np
from pathlib import Path
from spiketimes.simulate import homogeneous_poisson_process
from spiketimes.surrogates import shuffled_isi_spiketrain
from spiketimes.statistics import (
    mean_firing_rate,
    mean_firing_rate_ifr,
    inter_spike_intervals,
    spike_count_correlation,
    spike_count_correlation_test,
)


data_dir = Path(__file__).absolute().parent.parent / "data"
spiketrain_1_path = str(data_dir / "spiketrain_1.npy")
spiketrain_2_path = str(data_dir / "spiketrain_1.npy")


class TestMeanFiringRate:
    def test_expected(self):
        rate = 5
        num_secs = 120
        spiketrain = homogeneous_poisson_process(rate=rate, t_stop=num_secs)
        expected = rate
        actual = mean_firing_rate(spiketrain)
        assert pytest.approx(expected, actual)


class TestMeanFiringRateIFR:
    # add tests for min_fr
    def test_expected(self):
        rate = 10
        num_secs = 120
        fs = 1
        spiketrain = homogeneous_poisson_process(rate=rate, t_stop=num_secs)
        expected = rate
        actual = mean_firing_rate_ifr(spiketrain, fs=fs)
        assert pytest.approx(expected, actual)


class TestISI:
    def test_expected(self):
        spiketimes = np.arange(0, 10, 1)
        expected = np.ones((9, 1)).flatten()
        actual = inter_spike_intervals(spiketimes)
        np.testing.assert_allclose(actual, expected)


class TestSpikeCountCorr:
    def test_expected_same_spiketrain(self):
        spiketrain_1 = np.load(spiketrain_1_path)
        spiketrain_2 = np.load(spiketrain_1_path)
        fs = 1
        expected = 1
        actual = spike_count_correlation(spiketrain_1, spiketrain_2, fs=fs)
        pytest.approx(expected, actual)

    def test_expected_different_spiketrains(self):
        spiketrain_1 = np.load(spiketrain_1_path)
        spiketrain_2 = np.load(spiketrain_1_path)
        expected = 0.5455593433521561
        fs = 1
        actual = spike_count_correlation(spiketrain_1, spiketrain_2, fs=fs)
        pytest.approx(expected, actual)


class TestSpikeCountCorrT:
    def test_expected_correlated(self):
        spiketrain_1 = np.load(spiketrain_1_path)
        spiketrain_2 = np.load(spiketrain_1_path)
        expected_r = 0.5455593433521561
        expected_p = 0
        fs = 1
        r, p = spike_count_correlation_test(spiketrain_1, spiketrain_2, fs=fs)
        pytest.approx(expected_r, r)
        pytest.approx(expected_p, p)

    def test_uncorrelated(self):
        spiketrain_1 = np.load(spiketrain_1_path)
        spiketrain_2 = shuffled_isi_spiketrain(spiketrain_1)
        fs = 1

        _, p = spike_count_correlation_test(spiketrain_1, spiketrain_2, fs=fs)
        assert p > 0.01
