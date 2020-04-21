from .utils import _isi
from ..alignment import which_bin, align_to, binned_spiketrain
import numpy as np
from numpy.random import random


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
    isi: np.ndarray = _isi(spiketrain)
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


def jitter_spiketrain(
    spiketrain: np.ndarray,
    jitter_window_size: float,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Given a numpy array spiketrain, generates a single surrogate by jittering
    spiketimes. The spiketrain is discritised according to some jitter interval
    inside each bin the spike count should remain similar while the spiketimes
    are randomised.

    params:
        spiketrain: a numpy array of spiketimes in seconds
        jitter_window_size: the size of the timebins used to discretise the spiketrain
        t_start: if specified, spikes before this limit will be discarded
        t_stop: if specified, spikes after this limit will be discarded
    returns:
        a numpy array of a surrogate jittered spiketrain
    """
    if t_start is None:
        t_start = spiketrain[0]
    if t_stop is None:
        t_stop = spiketrain[-1]
    edges, values = binned_spiketrain(
        spiketrain, fs=(1 / jitter_window_size), t_start=t_start, t_stop=t_stop
    )
    jittered = np.sort(
        np.concatenate(
            [
                edge + (random(val) * jitter_window_size)
                for edge, val in zip(edges, values)
                if val != 0
            ]
        )
    )
    return jittered[jittered < t_stop]


def jitter_spiketrains(
    spiketrain: np.ndarray,
    n: int,
    jitter_window_size: float,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Given a numpy array spiketrain, generates n surrogates by jittering
    spiketimes. The spiketrain is discritised according to some jitter interval
    inside each bin the spike count should remain similar while the spiketimes
    are randomised.

    params:
        spiketrain: a numpy array of spiketimes in seconds
        n: the number of surrogate spiketrains to return 
        jitter_window_size: the size of the timebins used to discretise the spiketrain
        t_start: if specified, spikes before this limit will be discarded
        t_stop: if specified, spikes after this limit will be discarded
    returns:
        a list of numpy arrays containing surrogate jittered spiketrains
    """
    if t_start is None:
        t_start = spiketrain[0]
    if t_stop is None:
        t_stop = spiketrain[-1]
    return [
        jitter_spiketrain(
            spiketrain,
            jitter_window_size=jitter_window_size,
            t_start=t_start,
            t_stop=t_stop,
        )
        for _ in range(n)
    ]

