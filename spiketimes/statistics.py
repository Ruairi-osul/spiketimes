import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import variation, zmap
from sklearn.metrics import roc_auc_score
from .binning import binned_spiketrain
from mlxtend.evaluate import permutation_test


def ifr(
    spiketrain: np.ndarray,
    fs: float,
    t_start: float = None,
    t_stop: float = None,
    sigma: float = None,
    as_df: float = False,
):
    """
    Estimate instantaneous firing rate at a regular sampling rate.

    Args:
        spiketrain: A numpy array of spiketimes in seconds
        fs: The sampling rate at which to estimate firing rate
        t_start: If specified, only returns times after this point
        t_stop: If specified, only returns times before this point
        sigma: Parameter controling degree of smooting of firing rate estimates. Set to 0 for no smoothing.
        as_df: Whether to return results as pandas DataFrame
    Returns:
        time_bins, ifr
    """
    if len(spiketrain) <= 3:
        return np.nan

    if t_start is None:
        t_start = np.min(spiketrain)
    if t_stop is None:
        t_stop = np.max(spiketrain)
    df = binned_spiketrain(spiketrain, fs, t_stop=t_stop, t_start=t_start, as_df=True)
    df["spike_count"] = df["spike_count"].divide(1 / fs)

    # smoothing
    if sigma is None:
        # infer sigma
        sigma = _optimal_sigma(spike_counts=df["spike_count"].values)
    if sigma != 0:
        smoothed = gaussian_filter1d(df["spike_count"], sigma)
    else:
        # no smoothing
        smoothed = df["spike_count"].values

    if not as_df:
        return df["time"].values, smoothed
    else:
        return pd.DataFrame({"time": df["time"], "ifr": smoothed})


def _optimal_sigma(spike_counts):
    return 3 * np.std(spike_counts)


def mean_firing_rate(
    spiketrain: np.ndarray, t_start: float = None, t_stop: float = None
):
    """
    Calculate the mean firing rate of a spiketrain by summing total spikes
    and dividing by time.

    Args:
        spiketrain: A numpy array of spiketimes in seconds
        t_start: The start of the time over which mean firing rate will be calculated.
                 Defaults to the timepoint of the first spike
        t_end: The end of the time over which mean firing rate will be calculated
               defaults to the timepoint of the last spike
    Returns:
        The mean firing rate of the spiketrain
    """

    if len(spiketrain) <= 3:
        return np.nan
    if t_start is None:
        t_start = np.min(spiketrain)
    if t_stop is None:
        t_stop = np.max(spiketrain)

    total_spikes = len(spiketrain)
    total_time = t_stop - t_start

    return total_spikes / total_time


def mean_firing_rate_ifr(
    spiketrain: np.ndarray,
    fs: float,
    sigma: float = None,
    t_start: float = None,
    t_stop: float = None,
    exclude_below: float = None,
):
    """
    Calculate the mean firing rate of a spiketrain by first estimating the instantaneous
    firing rate at some sampling interval and then taking the median.

    Usefull when firing rate during active periods only is desired.

    Args:
        spiketrain: A numpy array of spiketimes in seconds
        fs: The sampling rate at which instantaneous rate is calculated
        sigma: A oarameter controlling the degree of smoothing level of firing rate estimates.
        t_start: The start of the time over which mean firing rate will be calculated.
                 Defaults to the timepoint of the first spike
        t_end: The end of the time over which mean firing rate will be calculated
               defaults to the timepoint of the last spike
        min_fr: If specified, calculates mean over time bins where mean firing rate is
                greater than this threshold
    Returns:
        A firing rate estiamte
    """
    if len(spiketrain) <= 3:
        return np.nan
    if t_start is None:
        t_start = np.min(spiketrain)
    if t_stop is None:
        t_stop = np.max(spiketrain)

    _, ifr_ = ifr(
        spiketrain, fs=fs, sigma=sigma, t_start=t_start, t_stop=t_stop, as_df=False
    )

    if exclude_below:
        ifr_ = ifr_[ifr_ >= exclude_below]

    return np.nanmedian(ifr_)


def inter_spike_intervals(spiketrain: np.ndarray):
    """
    Get the inter-spike-intervals of a spiketrain

    Args:
        spiketrain: a numpy array spike times
    Returns:
        A numpy array of inter spike intervals
    """
    return np.diff(np.sort(spiketrain))


def cov(arr: np.ndarray, axis: int = 0):
    """
    Computes the coefficient of variation.

    Simply wraps the scipy.stats variation function

    Args:
        arr: A numpy array
        axis: The axis over which to calculate cov
    Returns:
        The coefficient of variation
    """
    return variation(arr, axis=axis)


def cv2(arr: np.ndarray):
    """
    Compute the cv2 of an array.

    The Cv2 is a metric similar to the coefficient of variation but which includes
    a correction for signals which slowly fluctuate over time. [Suitable
    for long term neuronal recordings.]

    Args:
        arr: A numpy array on which to calculate cv2
    Returns:
        The cv2 of the array
    """
    return 2 * np.mean(np.absolute(np.diff(arr)) / (arr[:-1] + arr[1:]))


def cv2_isi(spiketrain: np.ndarray):
    """
    Calculate the cv2 of inter-spike-intervals.

    The Cv2 is a metric similar to the coefficient of variation but which includes
    a correction for signals which slowly fluctuate over time. [Suitable
    for long term neuronal recordings.]

    Args:
        spiketrain: a numpy array of spiketimes in seconds
    Returns:
        The cv2 of inter-spike-intervaks value
    """
    return cv2(inter_spike_intervals(spiketrain))


def cv_isi(spiketrain: np.ndarray):
    """
    Caluclate the coefficient of variation of inter-spike-intervals.

    Args:
        spiketrain: A numpy array of spiketimes
    Returns:
        The coeffient of variation of inter-spike-intervals
    """
    return cov(inter_spike_intervals(spiketrain))


def auc_roc(
    spike_counts: np.ndarray,
    which_condition: np.ndarray,
    return_distance_from_chance: bool = False,
):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve of spike counts from two conditions.

    The AUCROC can be used as a metric of the separability of two distrobutions.

    Args:
        spike_counts: A numpy array containing spike counts from both conditions
        which_condition: A numpy array indicating the condition of each spike count entry. 0s for the first condition,
                         and 1s for the second condition. For example, if the first two elements in
                         spike_counts were from the first condition and the third element from the
                         second condition, which_condition would contain [0, 0, 1]
        return_distance_from_chance: If True, returns distance from 0.5
    Returns:
        The AUCROC score
    """
    score = roc_auc_score(which_condition, spike_counts)
    if return_distance_from_chance:
        return abs(0.5 - score)
    else:
        return score


def zscore_standardise(to_standardise: np.ndarray, baseline: np.ndarray):
    """
    Convert an array to zscores calculated on a baseline period.

    Args:
        to_normalise: A numpy array to be converted to zscores.
        baseline: A numpy array containing data used to calculate the mean and standard deviation
                  for zscore conversions. This is usually (but not necessarily) a subsection of to_standardise
    Returns:
        A numpy array of zscores
    """
    return zmap(to_standardise, baseline)


def auc_roc_test(
    spike_counts: np.ndarray,
    which_condition: np.ndarray,
    n_boot: int = 1000,
    return_distance_from_chance: bool = False,
):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve of spike counts from two conditions.
    Also Test significance.

    The AUCROC can be used as a metric of the separability of two distrobutions.
    Significance tested using a permutation test.

    Args:
        spike_counts: A numpy array containing spike counts from both conditions
        which_condition: A numpy array indicating the condition of each spike count entry. 0s for the first condition,
                         and 1s for the second condition. For example, if the first two elements
                         in spike_counts were from the first condition and the third element from the second
                         condition, which_condition would contain [0, 0, 1]
        return_distance_from_chance: If True, returns distance from 0.5
    Returns:
        The AUCROC score, p
    """
    score = roc_auc_score(which_condition, spike_counts)
    if return_distance_from_chance:
        score = abs(0.5 - score)
    c1 = spike_counts[which_condition == 0]
    c2 = spike_counts[which_condition == 1]
    p = permutation_test(
        c1, c2, func=_aucroc_1d, method="approximate", num_rounds=n_boot,
    )
    return score, p


def _aucroc_1d(arr1, arr2, return_distance_from_chance=True):
    """
    Calculate AUC ROC on two arrays.
    """
    data = np.concatenate((arr1, arr2))
    y_true = np.concatenate((np.zeros(len(arr1)), np.ones(len(arr2))))
    score = roc_auc_score(y_true, data)
    if return_distance_from_chance:
        return abs(0.5 - score)
    else:
        return score
