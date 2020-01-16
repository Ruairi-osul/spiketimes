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


def jitter_spiketrains_old(
    spiketrain: np.ndarray,
    sampling_interval: float,
    n: int,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Given a numpy array of spiketimes, returns a list of spiketrains 
    with the same rate but with spiketimes jittered.

    params:
        spiketrain: np.ndarray of spiketimes
        sampling_interval: size of the window used to discretise and jitter the
                           spiketrain
        n: the number of spiketrains to return
        t_start: the left edge of the first time bin used to discretise the
                 spiketrain
        t_stop: the right edge of the last bin used to discretise the
                 spiketrain

    returns:
        a list of np.ndarrays of spiketimes 
    """
    if t_start is None:
        t_start = spiketrain[0]
    if t_stop is None:
        t_stop = spiketrain[-1]
    bin_edges = np.arange(t_start, t_stop, sampling_interval)
    _, bin_values = which_bin(spiketrain, bin_edges)
    offsets = align_to(spiketrain, bin_edges)

    random_shift_weights = random(
        size=(n, len(spiketrain))
    )  # numpy.random.random (linting correction)

    surrogates = np.sort(bin_values + (random_shift_weights * offsets), axis=1)
    return [s for s in surrogates]
