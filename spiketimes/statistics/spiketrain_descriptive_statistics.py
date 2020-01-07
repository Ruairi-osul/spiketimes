import numpy as np
from scipy.stats import variation


def inter_spike_intervals(spiketrain):
    """given an array of spike times, returns an 
    array of interspike intervals
    Params: 
        spiketrain: an array of spike times
    reutrns:
        a numpy array of inter spike intervals"""
    # TODO tests

    return np.diff(np.sort(spiketrain))

def cov(isi, axis=0):
    """Computes the coefficient of variation.
    Simply wraps the scipy.stats variation function
    """
    return variation(isi, axis=axis)