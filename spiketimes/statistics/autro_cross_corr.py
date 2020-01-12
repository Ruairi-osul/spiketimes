import numpy as np
import pandas as pd
from ..alignment import binned_spiketrain


def auto_corr(
    spiketrain: np.ndarray, fs: int = 100, num_lags: int = 100, as_df: bool = False
):
    """
    Given a spike train and a sampling rate, discretises the spiketrain
    and returns its autocorrelation. The autocorrelation array contains points
    (0 - num_lags) to (0 + num_lags) excluding 0 lag.

    params:
        spiketrain: a numpy array or pandas.Series containing 
                    timepoints of spiketimes
        fs: the sampling rate used to bin the spiketrain
        num_lags: the number of time bins to shift and correlate
    """

    _, bins = binned_spiketrain(spiketrain, fs=fs)
    bins = pd.Series(bins)
    time_bins = np.array([i for i in range(-num_lags, num_lags) if i != 0])
    auto_corr_ = np.array([bins.autocorr(i) for i in time_bins])
    if not as_df:
        return time_bins, auto_corr_
    else:
        return pd.DataFrame({"lag": time_bins, "auto_correlation": auto_corr_})


def cross_corr(
    spiketrain_1: np.ndarray,
    spiketrain_2: np.ndarray,
    fs: int = 100,
    num_lags: int = 100,
    as_df: bool = False,
):
    """
    Given two spiketrains and a sampling rate, discretises the spiketrains
    and return the cross correlation between spiketrain_1 and spiketrain_2. 
    This corresponds to the correlation between spiketrain_1 and spiketrain_2 
    shifted different time lags.
    The autocorrelation array contains points (0 - num_lags) to (0 + num_lags) 
    excluding 0 lag.

    params:
        spiketrain_1: a numpy array or pandas.Series containing 
                      timepoints of spiketimes
        spiketrain_2: a numpy array or pandas.Series containing 
                      timepoints of spiketimes
        fs: the sampling rate used to bin the spiketrain
        num_lags: the number of time bins to shift and correlate
        as_df: whether to return results as pandas DataFrame
    """
    t_start = np.min([np.min(spiketrain_1), np.min(spiketrain_2)])
    t_stop = np.max([np.max(spiketrain_1), np.max(spiketrain_2)])

    _, bins_1 = binned_spiketrain(spiketrain_1, fs=fs, t_start=t_start, t_stop=t_stop)
    _, bins_2 = binned_spiketrain(spiketrain_2, fs=fs, t_start=t_start, t_stop=t_stop)

    bins_1 = pd.Series(bins_1)
    bins_2 = pd.Series(bins_2)

    time_bins = np.array([i for i in range(-num_lags, num_lags) if i != 0])

    cross_corr_ = np.array([bins_1.corr(bins_2.shift(i)) for i in time_bins])
    if not as_df:
        return time_bins, cross_corr_
    else:
        return pd.DataFrame({"lag": time_bins, "cross_correlation": cross_corr_})
