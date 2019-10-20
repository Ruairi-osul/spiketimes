from spiketimes.simulate import homogenous_poisson_process, imhomogenous_poisson_process
import numpy as np
import pytest


class TestHomogenous:
    def test_num_output_samples(self):
        rate = 4
        n_time_units = 30
        n_bootstrap_replicates = 1000
        expected = int(rate * n_time_units)
        actual = int(
            np.mean(
                np.array(
                    [
                        len(homogenous_poisson_process(rate, n_time_units))
                        for _ in range(n_bootstrap_replicates)
                    ]
                )
            )
        )
        assert np.absolute(actual - expected) < 5

    def test_output_shape(self):
        expected = 1
        actual = int(len(homogenous_poisson_process(4, 10, 0).shape))
        assert actual == expected


class TestImhomogenous:
    def test_num_output_samples(self):
        time_1 = 30
        time_2 = 20
        rate_1 = 2
        rate_2 = 3
        time_rate = [(time_1, rate_1), (time_2, rate_2)]
        n_bootstrap_replicates = 1000
        expected = int((time_1 * rate_1) + (time_2 * rate_2))
        actual = int(
            np.mean(
                np.array(
                    [
                        len(imhomogenous_poisson_process(time_rate))
                        for _ in range(n_bootstrap_replicates)
                    ]
                )
            )
        )
        assert np.absolute(actual - expected) < 5

    def test_output_shape(self):
        expected = 1
        actual = len(imhomogenous_poisson_process([(1, 2), (2, 1)]).shape)
        assert actual == expected
