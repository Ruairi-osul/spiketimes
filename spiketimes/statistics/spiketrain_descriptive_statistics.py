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


def cov(arr: np.ndarray, axis: int = 0):
    """
    Computes the coefficient of variation.
    Simply wraps the scipy.stats variation function

    params:
        arr: a numpy array
        axis: the axis over which to calculate cov
    returns:
        cov value
    """
    return variation(arr, axis=axis)


def cv2(arr: np.ndarray):
    """
    A metric similar to the coefficient of variation but which includes
    a correction for signals which slowly fluctuate over time. [Suitable
    for long term neuronal recordings.]

    params:
        arr: numpy array on which to calculate cv2
    returns:
        cv2 value 
    """
    return 2 * np.mean(np.absolute(np.diff(arr)) / (arr[:-1] + arr[1:]))


def cv2_isi(spiketrain: np.ndarray):
    """
    Given a numpy array of spiketimes, calculates the cv2 of its 
    inter-spike-intervals. This is a metric of the regularity at which the
    neuron fired.

    params:
        spiketrain: a numpy array of spiketimes in seconds
    returns:
        cv2_isi value
    """
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
