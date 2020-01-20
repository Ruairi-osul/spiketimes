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


def cov(isi: np.ndarray, axis: int = 0):
    """Computes the coefficient of variation.
    Simply wraps the scipy.stats variation function
    """
    return variation(isi, axis=axis)


def cv2(isi: np.ndarray):
    return 2 * np.mean(np.absolute(np.diff(isi)) / (isi[:-1] + isi[1:]))


def cv2_isi(spiketrain: np.ndarray):
    return cv2(inter_spike_intervals(spiketrain))


def cv_isi(spiketrain: np.ndarray):
    """
    given an array of spike times, calculates the coefficient of 
    variation of interspike intervals
    Params: 
        spiketrain: an array of spike times
    reutrns:
        the coeffient of variation
    """
    return cov(inter_spike_intervals(spiketrain))
