import pytest
from spiketimes.simulate import homogenous_poisson_process
from spiketimes.statistics import mean_firing_rate, mean_firing_rate_ifr


class TestMeanFiringRate:
    def test_expected(self):
        rate = 5
        num_secs = 120
        spiketrain = homogenous_poisson_process(rate=rate, t_stop=num_secs)
        expected = rate
        actual = mean_firing_rate(spiketrain)
        assert pytest.approx(expected, actual)


class TestMeanFiringRateIFR:
    # add tests for min_fr
    def test_expected(self):
        rate = 10
        num_secs = 120
        fs = 1
        spiketrain = homogenous_poisson_process(rate=rate, t_stop=num_secs)
        expected = rate
        actual = mean_firing_rate_ifr(spiketrain, fs=fs)
        assert pytest.approx(expected, actual)
