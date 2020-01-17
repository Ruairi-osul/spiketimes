import numpy as np
import pandas as pd
from scipy import signal
from ..alignment import binned_spiketrain
from ..surrogates import jitter_spiketrains
from .utils import _random_combination, p_adjust


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
    jitter_window_size: float = 0.4,
    n_boot: int = 4000,
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
    at each bin. Significance is tested by comparing observed crosscorrelation values
    to bootstrap replicates. Replicates are generated by calculating surrogate 
    jitter spiketrains. p values may be adjusted for multiple comparisons using a 
    variety of methods.

    params:
        spiketrain_1: nd array of spiketimes in seconds
        spiketrain_1: nd array of spiketimes in seconds
        bin_window: size of bins in seconds used to discretise the spiketrain
        jitter_window_size: size of jittering window
        n_boot: number of bootstrap replicate

    """

    if t_start is None:
        t_start = np.nanmin([np.nanmin(spiketrain_1), np.nanmin(spiketrain_2)])
    if t_stop is None:
        t_stop = np.nanmax([np.nanmax(spiketrain_1), np.nanmax(spiketrain_2)])

    # get observed
    time_bins, observed_cc = cross_corr(
        spiketrain_1,
        spiketrain_2,
        bin_window=bin_window,
        num_lags=num_lags,
        as_df=False,
        t_start=t_start,
        t_stop=t_stop,
    )

    # get surrogates
    n_surrogates = n_boot // 3
    st_1_surrogates = jitter_spiketrains(
        spiketrain_1,
        jitter_window_size=jitter_window_size,
        t_start=t_start,
        t_stop=t_stop,
        n=n_surrogates,
    )
    st_2_surrogates = jitter_spiketrains(
        spiketrain_2,
        jitter_window_size=jitter_window_size,
        t_start=t_start,
        t_stop=t_stop,
        n=n_surrogates,
    )
    surrogate_indexes = list(range(n_surrogates))

    # generate replicates
    replicates = []
    for _ in range(n_boot):
        surrogate_pair = _random_combination(surrogate_indexes, r=2)
        neuron_1, neuron_2 = (
            st_1_surrogates[surrogate_pair[0]],
            st_2_surrogates[surrogate_pair[1]],
        )
        replicates.append(
            cross_corr(
                neuron_1,
                neuron_2,
                bin_window=bin_window,
                num_lags=num_lags,
                as_df=False,
                t_start=t_start,
                t_stop=t_stop,
            )[1]
        )
    replicates = np.array(replicates)

    p = []
    if tail == "two_tailed":
        replicates = np.absolute(replicates)
        for i, observed in enumerate(observed_cc):
            reps = replicates[:, i]
            if observed < np.nanmean(reps):
                p.append(np.nanmean(reps < observed))
            else:
                p.append(np.nanmean(reps > observed))
    elif tail == "upper":
        p = [
            np.nanmean(reps[:, i] > observed for i, observed in enumerate(observed_cc))
        ]
    elif tail == "lower":
        p = [
            np.nanmean(reps[:, i] < observed for i, observed in enumerate(observed_cc))
        ]
    else:
        raise ValueError(
            "Could not parse tail value. Select one of"
            "{'Two tailed', 'lower', 'upper'} - 'upper' if hypothesising a positive r"
        )
    p = np.array(p)
    if tail == "two_tailed":
        p = p * 2

    if adjust_p:
        p = p_adjust(p, method=p_adjust_method)

    if not as_df:
        return time_bins, observed_cc, p
    else:
        return pd.DataFrame(
            {"time_sec": time_bins, "crosscorrelation": observed_cc, "p": p}
        )
