import numpy as np
from scipy.stats import variation


def inter_spike_intervals(spiketrain: np.ndarray):
    """
    given an array of spike times, returns an 
    array of interspike intervals
    Params: 
        spiketrain: an array of spike times
    reutrns:
        a numpy array of inter spike intervals"""

    return np.diff(np.sort(spiketrain))


def cov(isi: np.ndarray, axis=0):
    """Computes the coefficient of variation.
    Simply wraps the scipy.stats variation function
    """
    return variation(isi, axis=axis)


def cv_isi(spiketrain: np.ndarray):
    """
    given an array of spike times, calculates the coefficient of 
    variation of interspike intervals
    Params: 
        spiketrain: an array of spike times
    reutrns:
        the coeffient of variation (np.float64)
    """
    return cov(inter_spike_intervals(spiketrain))
