import numpy as np
from ..statistics import inter_spike_intervals


def shuffled_isi_spiketrain(spiketrain: np.ndarray):
    """
    Given a numpy array of spiketimes, returns another with
    the same interspike intervals but shuffled

    params:
        spiketrain: np.ndarray of spiketimes

    returns:
        np.array of spiketimes 
    """
    spiketrain = np.copy(spiketrain)
    isi = inter_spike_intervals(spiketrain)
    np.random.shuffle(isi)
    return np.cumsum(isi)


def shuffled_isi_spiketrains(spiketrain: np.ndarray, n: int = 1):
    """
    Given a numpy array of spiketimes, returns a list of spiketrains 
    with the same interspike intervals but shuffled

    params:
        spiketrain: np.ndarray of spiketimes

    returns:
        a list of np.array of spiketimes 
    """
    return [shuffled_isi_spiketrain(spiketrain) for _ in range(n)]
