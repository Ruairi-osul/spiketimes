from .utils import _isi
from .binning import binned_spiketrain
import numpy as np
from numpy.random import random


def shuffled_isi_spiketrain(spiketrain: np.ndarray):
    """
    Return a surrogate spiketrain with shuffled inter spike intervals.

    Args:
        spiketrain: The parent array spiketrain from which to create the surrogate
    Returns:
        A numpy array spiketrain containing spiketimes in seconds
    """
    spiketrain = np.copy(spiketrain)
    t_start = np.min(spiketrain)
    isi: np.ndarray = _isi(spiketrain)
    np.random.shuffle(isi)
    return np.cumsum(isi) + t_start


def shuffled_isi_spiketrains(spiketrain: np.ndarray, n: int = 1):
    """
    Return n surrogate spiketrains with shuffled inter spike intervals.

    Args:
        spiketrain: The parent array spiketrain from which to create the surrogate
        n: The number of surrogate spiketrains to return
    Returns:
        A list of numpy array spiketrain containing spiketimes in seconds
    """
    return [shuffled_isi_spiketrain(spiketrain) for _ in range(n)]


def jitter_spiketrain(
    spiketrain: np.ndarray,
    jitter_window_size: float,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Return a jitter spiketrain surrogate from a parent spiketrain.

    Jitter spiketrains contain similar firing rate dynamics to the parent spiketrain but with spiketimes
    randomised. This is done by binning spikecounts over time and then randomising spiketimes within
    time bins.

    Args:
        spiketrain: The parent numpy spiketrain.
        jitter_window_size: The size of the timebins in seconds
        t_start: If specified, spikes before this limit will be discarded
        t_stop: If specified, spikes after this limit will be discarded
    Returns:
        A numpy array spiketrain containing spiketimes in seconds
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
    Return a jitter spiketrain surrogate from a parent spiketrain.

    Jitter spiketrains contain similar firing rate dynamics to the parent spiketrain but with spiketimes
    randomised. This is done by binning spikecounts over time and then randomising spiketimes within
    time bins.

    Args:
        spiketrain: The parent numpy spiketrain.
        jitter_window_size: The size of the timebins in seconds
        t_start: If specified, spikes before this limit will be discarded
        t_stop: If specified, spikes after this limit will be discarded
    Returns:
        A list of numpy array spiketrain containing spiketimes in seconds
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
