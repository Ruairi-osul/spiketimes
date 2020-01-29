import numpy as np
import pandas as pd
from scipy import signal
from ..alignment import binned_spiketrain
from ..surrogates import jitter_spiketrains
from .utils import _random_combination, p_adjust, _ppois


def auto_corr(
    spiketrain: np.ndarray,
    bin_window: float = 0.01,
    num_lags: int = 100,
    as_df: bool = False,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Given a spike train and a sampling rate, discretises the spiketrain
    and returns its autocorrelation. The autocorrelation array contains points
    (0 - num_lags) to (0 + num_lags) excluding 0 lag.

    params:
        spiketrain: a numpy array or pandas.Series containing 
                    timepoints of spiketimes
        bin_window: the size of the bins used to discretise the spiketrain
        num_lags: the number of time bins to shift and correlate
        as_df: whether to return results as pandas DataFrame
        t_start: if specified, provides the left edge of the first time bin used
                 to discretise the spiketrain
        t_stop: if specified, provides the right edge of the last time bin used
                 to discretise the spiketrain
    returns:
        time_bins, autocorrelation_values
    """

    # get lag labels
    time_span = bin_window * num_lags
    time_bins = np.arange(-time_span, time_span + bin_window, bin_window)
    time_bins = np.delete(time_bins, len(time_bins) // 2)  # delete 0 element

    # discretise the spiketrain
    if t_start is None:
        t_start = np.nanmin(spiketrain)
    if t_stop is None:
        t_stop = np.nanmax(spiketrain)
    _, binned_spiketrain_ = binned_spiketrain(
        spiketrain, fs=(1 / bin_window), t_start=t_start, t_stop=t_stop
    )

    # get autocorrelation values
    vals = signal.correlate(binned_spiketrain_, binned_spiketrain_, mode="same")
    zero_idx = len(vals) // 2
    vals = vals[(zero_idx - num_lags) : (zero_idx + num_lags + 1)]
    vals = np.delete(vals, len(vals) // 2)  # delete 0 element

    if not as_df:
        return time_bins, vals
    else:
        return pd.DataFrame({"time_sec": time_bins, "autocorrelation": vals})


def cross_corr(
    spiketrain_1: np.ndarray,
    spiketrain_2: np.ndarray,
    bin_window: float = 0.01,
    num_lags: int = 100,
    as_df: bool = False,
    t_start: float = None,
    t_stop: float = None,
    delete_0_lag: bool = False,
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
        bin_window: the size of the bins used to discretise the spiketrain
        num_lags: the number of time bins to shift and correlate
        as_df: whether to return results as pandas DataFrame
        t_start: if specified, provides the left edge of the first time bin used
                 to discretise the spiketrain
        t_stop: if specified, provides the right edge of the last time bin used
                 to discretise the spiketrain
    returns:
        time_bins, crosscorrelation_values
    """

    # get lag labels
    time_span = bin_window * num_lags
    time_bins = np.arange(-time_span, time_span + bin_window, bin_window)

    # discretise the spiketrain
    if t_start is None:
        t_start = np.nanmin([np.nanmin(spiketrain_1), np.nanmin(spiketrain_2)])
    if t_stop is None:
        t_stop = np.nanmax([np.nanmax(spiketrain_1), np.nanmax(spiketrain_2)])

    _, bins_1 = binned_spiketrain(
        spiketrain_1, fs=(1 / bin_window), t_start=t_start, t_stop=t_stop
    )
    _, bins_2 = binned_spiketrain(
        spiketrain_2, fs=(1 / bin_window), t_start=t_start, t_stop=t_stop
    )

    # get crosscorrelation values
    vals = signal.correlate(bins_1, bins_2, mode="same")
    zero_idx = len(vals) // 2
    vals = vals[(zero_idx - num_lags) : (zero_idx + num_lags + 1)]

    if delete_0_lag:
        time_bins = np.delete(time_bins, len(time_bins) // 2)
        vals = np.delete(vals, len(vals) // 2)

    if not as_df:
        return time_bins, vals
    else:
        return pd.DataFrame({"time_sec": time_bins, "crosscorrelation": vals})


def cross_corr_test(
    spiketrain_1: np.ndarray,
    spiketrain_2: np.ndarray,
    bin_window: float = 0.01,
    num_lags: int = 100,
    as_df: bool = False,
    t_start: float = None,
    t_stop: float = None,
    tail: str = "two_tailed",
    adjust_p: bool = True,
    p_adjust_method: str = "Benjamini-Hochberg",
):
    """
    Calculate the cross correlation between two neurons and test the significance 
    at each bin. Significance is tested by testing observed crosscorrelation to an expected
    distrobution under poisson assumptions. p values may (and should) be adjusted for multiple 
    comparisons using a variety of methods.

    params:
        spiketrain_1: nd array of spiketimes in seconds
        spiketrain_1: nd array of spiketimes in seconds
        bin_window: size of bins in seconds used to discretise the spiketrain
        num_lags: number of lags to return. If x, then 0 - x to 0 + x bins are returned
        as_df: whether to return the results as a pandas DataFrame
        t_start: if specified, only spikes after this value will be included in the calculation
        t_stop: if specified, only spikes before this value will be included in the calculation
        tail: for hypothesis tests, whether to perform lower, upper or two_tailled tests
        adjust_p: whether to adjust p values for multiple comparisons
        p_adjust_method: method to use for adjusting p values. Must be in the following set:
                         {'Bonferroni', 'Bonferroni-Holm', 'Benjamini-Hochberg'}
    returns:
        timebins, crosscorrelation_values, p_values
    """

    if t_start is None:
        t_start = np.nanmin([np.nanmin(spiketrain_1), np.nanmin(spiketrain_2)])
    if t_stop is None:
        t_stop = np.nanmax([np.nanmax(spiketrain_1), np.nanmax(spiketrain_2)])

    # get observed
    t, cc = cross_corr(
        spiketrain_1,
        spiketrain_2,
        bin_window=bin_window,
        num_lags=num_lags,
        as_df=False,
        t_start=t_start,
        t_stop=t_stop,
    )
    lam = np.mean(cc)
    p = np.array(list(map(lambda x: _ppois(x, mu=lam, tail="two_tailed"), cc)))
    p = np.array(p)

    if tail == "two_tailed":
        p = p * 2

    if adjust_p:
        p = p_adjust(p, method=p_adjust_method)

    if not as_df:
        return t, cc, p
    else:
        return pd.DataFrame({"time_sec": t, "crosscorrelation": cc, "p": p})
