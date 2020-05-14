import numpy as np
import pandas as pd
from scipy import stats
from scipy import signal
from .binning import binned_spiketrain
from .utils import p_adjust, _ppois
from .statistics import ifr
from .surrogates import shuffled_isi_spiketrains
from .utils import _random_combination


def auto_corr(
    spiketrain: np.ndarray,
    binsize: float = 0.01,
    num_lags: int = 100,
    as_df: bool = False,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Returns the autocorrelation function of a spiketrain.

    Args:
        spiketrain: A numpy array of spiketimes
        binsize: The size of the time bin in seconds
        num_lags: The number of lags forward and backwards around lag 0 to return
        t_start: Minimum timepoint
        t_stop: Maximum timepoint
        as_df: Whether to return results as pandas DataFrame
    Returns:
        time_bins, autocorrelation_values
    """

    # get lag labels
    time_span = binsize * num_lags
    time_bins = np.arange(-time_span, time_span + binsize, binsize)
    time_bins = np.delete(time_bins, len(time_bins) // 2)  # delete 0 element

    # discretise the spiketrain
    if t_start is None:
        t_start = np.nanmin(spiketrain)
    if t_stop is None:
        t_stop = np.nanmax(spiketrain)
    _, binned_spiketrain_ = binned_spiketrain(
        spiketrain, fs=(1 / binsize), t_start=t_start, t_stop=t_stop
    )

    # get autocorrelation values
    vals = signal.correlate(binned_spiketrain_, binned_spiketrain_, mode="same")
    zero_idx = len(vals) // 2
    vals = vals[(zero_idx - num_lags) : (zero_idx + num_lags + 1)]
    vals = np.delete(vals, len(vals) // 2)  # delete 0 element

    if not as_df:
        return time_bins, vals
    else:
        return pd.DataFrame({"time_bin": time_bins, "autocorrelation": vals})


def cross_corr(
    spiketrain_1: np.ndarray,
    spiketrain_2: np.ndarray,
    binsize: float = 0.01,
    num_lags: int = 100,
    as_df: bool = False,
    t_start: float = None,
    t_stop: float = None,
    delete_0_lag: bool = False,
):
    """
    Calculate crosscorrelation between two spiketrains.

    Args:
        spiketrain_1: A numpy array of spiketimes.
        spiketrain_2: A numpy array of spiketimes
        binsize: The size of the time bin in seconds
        num_lags: The number of lags forward and backwards around lag 0 to return
        as_df: Whether to return results as pandas DataFrame
        t_start: Minimum timepoint
        t_stop: Maximum timepoint
        delete_0_lag: Wheter to remove the 0-lag element
    Returns:
        time_bins, crosscorrelation_values
    """
    # get lag labels
    time_span = binsize * num_lags
    time_bins = np.arange(-time_span, time_span + binsize, binsize)

    # discretise the spiketrain
    if t_start is None:
        t_start = np.nanmin([np.nanmin(spiketrain_1), np.nanmin(spiketrain_2)])
    if t_stop is None:
        t_stop = np.nanmax([np.nanmax(spiketrain_1), np.nanmax(spiketrain_2)])

    _, bins_1 = binned_spiketrain(
        spiketrain_1, fs=(1 / binsize), t_start=t_start, t_stop=t_stop
    )
    _, bins_2 = binned_spiketrain(
        spiketrain_2, fs=(1 / binsize), t_start=t_start, t_stop=t_stop
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
        return pd.DataFrame({"time_bin": time_bins, "crosscorrelation": vals})


def cross_corr_test(
    spiketrain_1: np.ndarray,
    spiketrain_2: np.ndarray,
    binsize: float = 0.01,
    num_lags: int = 100,
    as_df: bool = False,
    t_start: float = None,
    t_stop: float = None,
    tail: str = "two_tailed",
    adjust_p: bool = True,
    p_adjust_method: str = "Benjamini-Hochberg",
):
    """
    Calculate crosscorrelation between two spiketrains. Also test significance of each bin.

    Significance test performed by comparing observed crosscorrelation to expected cross correlation of
    poisson spiketrains.

    Args:
        spiketrain_1: A numpy array of spiketimes.
        spiketrain_2: A numpy array of spiketimes
        binsize: The size of the time bin in seconds
        num_lags: The number of lags forward and backwards around lag 0 to return
        as_df: Whether to return results as pandas DataFrame
        t_start: Minimum timepoint
        t_stop: Maximum timepoint
        delete_0_lag: Wheter to remove the 0-lag element
        tail: Tail for hypothesis test {"two_tailed", "upper", "lower"}. Two tailed reccomended
        adjust_p: Whether to adjust p-values for multiple comparisons.
        p_adjust_method: If adjusting p-values, specified which method to use {Benjamini-Hochberg', 'Bonferroni',
                        'Bonferroni-Holm'}
    Returns:
        time_bins, crosscorrelation_values, p
    """

    if t_start is None:
        t_start = np.nanmin([np.nanmin(spiketrain_1), np.nanmin(spiketrain_2)])
    if t_stop is None:
        t_stop = np.nanmax([np.nanmax(spiketrain_1), np.nanmax(spiketrain_2)])

    # get observed
    t, cc = cross_corr(
        spiketrain_1,
        spiketrain_2,
        binsize=binsize,
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
        return pd.DataFrame({"time_bin": t, "crosscorrelation": cc, "p": p})


def spike_count_correlation(
    spiketrain_1: np.ndarray,
    spiketrain_2: np.ndarray,
    binsize: float,
    min_firing_rate: float = None,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Calculate spike count correlation between two spiketrains.

    Args:
        spiketrain_1: A numpy array of spiketimes.
        spiketrain_2: A numpy array of spiketimes
        binsize: The size of the time bin in seconds
        min_firing_rate: If selected, selects only bins where the geometric mean
                         firing rate of the two spiketrains exeedes this value.
        t_start: Minimum timepoint
        t_stop: Maximum timepoint
    Returns:
        Pearson's correlation coefficient
    """

    if t_start is None:
        t_start = np.min([np.min(spiketrain_1), np.min(spiketrain_2)])
    if t_stop is None:
        t_stop = np.max([np.max(spiketrain_1), np.max(spiketrain_2)])

    # convert to binned spike train
    _, st1_binned = binned_spiketrain(
        spiketrain_1, fs=(1 / binsize), t_start=t_start, t_stop=t_stop
    )
    _, st2_binned = binned_spiketrain(
        spiketrain_2, fs=(1 / binsize), t_start=t_start, t_stop=t_stop
    )

    if not min_firing_rate:
        return np.corrcoef(st1_binned, st2_binned)[0, 1]

    ifr_1 = ifr(spiketrain_1, fs=(1 / binsize), t_start=t_start, t_stop=t_stop).rename(
        columns={"ifr": "ifr_1"}
    )
    ifr_2 = ifr(spiketrain_2, fs=(1 / binsize), t_start=t_start, t_stop=t_stop).rename(
        columns={"ifr": "ifr_2"}
    )
    df = pd.merge(ifr_1, ifr_2).assign(st1_binned=st1_binned, st2_binned=st2_binned)
    df["gmean"] = stats.gmean(df.loc[:, ["ifr_1", "ifr_2"]], axis=1)
    df = df.copy().loc[df["gmean"] > min_firing_rate]
    if len(df) == 0:
        return np.nan
    return np.corrcoef(df["st1_binned"], df["st2_binned"])[0, 1]


def spike_count_correlation_test(
    spiketrain_1: np.ndarray,
    spiketrain_2: np.ndarray,
    binsize: float,
    n_boot: int = 500,
    min_firing_rate: float = None,
    t_start: float = None,
    t_stop: float = None,
    tail: str = "two_tailed",
):
    """
    Calculate peason's correlation coefficient between spikecounts of a pair of spiketrains.

    Args:
        spiketrain_1: The first spiketrain to correlate.
                      Should be an nd.array of spiketimes in seconds.
        spiketrain_2: The second spiketrain to correlate.
                      Should be an nd.array of spiketimes in seconds.
        binsize: The size of time bins used in seconds.
        min_firing_rate: If selected, selects only bins where the geometric mean
                         firing rate of the two spiketrains exeedes this value
        t_start: The startpoint of the first bin. Defaults to the first spike in the two trains
        t_stop: The maximum time for a time bin. Defaults to the last spike in the two trians
    Returns:
        R, p
    """
    r = spike_count_correlation(spiketrain_1, spiketrain_2, binsize=binsize)
    n_surrogates = int(n_boot / 3)
    st1_shuffled = shuffled_isi_spiketrains(spiketrain_1, n=n_surrogates)
    st2_shuffled = shuffled_isi_spiketrains(spiketrain_2, n=n_surrogates)
    combs = np.array(
        [_random_combination(np.arange(n_surrogates), r=2) for _ in range(n_boot)],
        dtype=np.int64,
    )
    replicates = np.apply_along_axis(
        lambda x: spike_count_correlation(
            st1_shuffled[x[0]], st2_shuffled[x[1]], binsize=binsize
        ),
        arr=combs,
        axis=1,
    )
    if tail == "two_tailed":
        replicates = np.absolute(replicates)
        p = np.nanmean(replicates >= np.absolute(r)) * 2
    elif tail == "upper":
        p = np.nanmean(replicates >= p)
    elif tail == "lower":
        p = np.nanmean(replicates <= p)
    else:
        raise ValueError(
            "Could not parse tail value. Select one of"
            "{'Two tailed', 'lower', 'upper'} - 'upper' if hypothesising a positive r"
        )
    return r, p
