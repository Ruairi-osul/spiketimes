from ..alignment import binned_spiketrain
from elephant.statistics import sskernel
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import numpy as np


def ifr(
    spiketrain: np.ndarray,
    fs: float,
    t_start: float = None,
    t_stop: float = None,
    sigma: float = None,
    as_df: float = True,
):
    """
    Given a numpy array of spiketimes, estimates its instantaneous firing rate.
    Firing rate is estimated by discretising the spiketrain and convolving with
    a gaussian kernal.

    params:
        spiketrain: numpy array of spiketimes in seconds
        fs: sampling rate to use to discretise the spiketrain
        t_start: if specified, only returns times after this point
        t_stop: it specified, only returns times before this point
        sigma: hyperparameter used to control width of gaussian kernel.
               if left as None, will guess appropriate shape based on spiking 
               properties
        as_df: whether to return results as pandas DataFrame
    returns:
        time_points, ifr

    """
    if t_start is None:
        t_start = spiketrain[0]
    if t_stop is None:
        t_stop = spiketrain[-1]
    df = binned_spiketrain(spiketrain, fs, t_stop=t_stop, t_start=t_start, as_df=True)
    df["spike_count"] = df["spike_count"].divide(1 / fs)
    if sigma is None:
        sigma = sskernel(spiketrain, tin=None, bootstrap=False)["optw"]
    smoothed = gaussian_filter1d(df["spike_count"], sigma)
    if not as_df:
        return df["time"].values, smoothed
    else:
        return pd.DataFrame({"time": df["time"], "ifr": smoothed})


def mean_firing_rate(
    spiketrain: np.ndarray, t_start: float = None, t_stop: float = None
):
    """
    Calculate the mean firing rate of a spiketrain by summing total spikes
    and dividing by time.

    params:
        spiketrain: numpy array of spiketimes in seconds
        t_start: the start of the time over which mean firing rate will be calculated
                 defaults to the timepoint of the first spike
        t_end: the end of the time over which mean firing rate will be calculated
               defaults to the timepoint of the last spike
    returns:
        mean firing rate
    """
    if t_start is None:
        t_start = spiketrain[0]
    if t_stop is None:
        t_stop = spiketrain[-1]

    total_spikes = len(spiketrain)
    total_time = t_start - t_stop

    return total_spikes / total_time


def mean_firing_rate_ifr(
    spiketrain: np.ndarray,
    fs: float,
    t_start: float = None,
    t_stop: float = None,
    min_fr: float = None,
):
    """
    Calculate the mean firing rate of a spiketrain by first estimating the instantaneous
    firing rate at some sampling interval and then taking the mean.

    params:
        spiketrain: a numpy array of spiketimes in seconds
        fs: the sampling rate at which instantaneous rate is calculated
        t_start: the start of the time over which mean firing rate will be calculated
                 defaults to the timepoint of the first spike
        t_end: the end of the time over which mean firing rate will be calculated
               defaults to the timepoint of the last spike
        min_fr: if specified, calculates mean over time bins where mean firing rate is
                greater than this threshold
    returns:
        mean_firing_rate
    """
    if t_start is None:
        t_start = spiketrain[0]
    if t_stop is None:
        t_stop = spiketrain[-1]

    _, ifr_ = ifr(spiketrain, fs=fs, t_start=t_start, t_stop=t_stop, as_df=False)

    if min_fr:
        ifr_ = ifr_[ifr_ > min_fr]

    return np.mean(ifr_)

