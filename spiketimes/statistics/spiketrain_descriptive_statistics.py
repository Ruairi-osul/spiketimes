import numpy as np


def inter_spike_intervals(spiketrain):
    """given an array of spike times, returns an 
    array of interspike intervals
    
    Params: 
        spiketrain: an array of spike times
    
    reutrns:
        a numpy array of inter spike intervals"""
    # TODO tests

    return np.diff(np.sort(spiketrain))
