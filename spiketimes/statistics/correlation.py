import numpy as np
import pandas as pd
from scipy import stats
from ..alignment import binned_spiketrain
from ..statistics import ifr
from ..surrogates import shuffled_isi_spiketrains
from .utils import _random_combination


def spike_count_correlation_test(
    spiketrain_1: np.ndarray,
    spiketrain_2: np.ndarray,
    fs: int,
    n_boot: int = 500,
    min_firing_rate: float = None,
    t_start: float = None,
    t_stop: float = None,
    tail: str = "two_tailed",
):
    """
    Given two numpy arrays of spiketimes and some sampling rate, calculates
    the pearson correlation coeficient for those neurons and calculates a p value
    by shuffling generating surrogate spiketrains with shuffled interspike intervals

    params:
        spiketrain_1: The first spiketrain to correlate. 
                      Should be an nd.array of spiketimes in seconds.
        spiketrain_2: The second spiketrain to correlate. 
                      Should be an nd.array of spiketimes in seconds.
        fs: The sampling rate use to bin the spikes before correlating
        min_firing_rate: If selected, selects only bins where the geometric mean
                         firing rate of the two spiketrains exeedes this value
        t_start: The startpoint of the first bin. Defaults to the first spike in the two trains
        t_stop: The maximum time for a time bin. Defaults to the last spike in the two trians
    
    returns:
        pearson's r
        p value
    """
    r = spike_count_correlation(spiketrain_1, spiketrain_2, fs=fs)
    n_surrogates = int(n_boot / 3)
    st1_shuffled = shuffled_isi_spiketrains(spiketrain_1, n=n_surrogates)
    st2_shuffled = shuffled_isi_spiketrains(spiketrain_2, n=n_surrogates)
    combs = np.array(
        [_random_combination(np.arange(n_surrogates), r=2) for _ in range(n_boot)],
        dtype=np.int64,
    )
    replicates = np.apply_along_axis(
        lambda x: spike_count_correlation(st1_shuffled[x[0]], st2_shuffled[x[1]], fs=1),
        arr=combs,
        axis=1,
    )
    if tail == "two_tailed":
        replicates = np.absolute(replicates)
        p = np.nanmean(replicates >= np.absolute(r))
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


def spike_count_correlation(
    spiketrain_1: np.ndarray,
    spiketrain_2: np.ndarray,
    fs: int,
    min_firing_rate: float = None,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Given two numpy arrays of spiketimes and some sampling rate, calculates
    the pearson correlation coeficient for those neurons by binning spikes into a specifed 
    bin size

    params:
        spiketrain_1: The first spiketrain to correlate. 
                      Should be an nd.array of spiketimes in seconds.
        spiketrain_2: The second spiketrain to correlate. 
                      Should be an nd.array of spiketimes in seconds.
        fs: The sampling rate use to bin the spikes before correlating
        min_firing_rate: If selected, selects only bins where the geometric mean
                         firing rate of the two spiketrains exeedes this value
        t_start: The startpoint of the first bin. Defaults to the first spike in the two trains
        t_stop: The maximum time for a time bin. Defaults to the last spike in the two trians
    
    returns:
        pearson's r 
    """

    if t_start is None:
        t_start = np.min([np.min(spiketrain_1), np.min(spiketrain_2)])
    if t_stop is None:
        t_stop = np.max([np.max(spiketrain_1), np.max(spiketrain_2)])

    # convert to binned spike train
    _, st1_binned = binned_spiketrain(
        spiketrain_1, fs=fs, t_start=t_start, t_stop=t_stop
    )
    _, st2_binned = binned_spiketrain(
        spiketrain_2, fs=fs, t_start=t_start, t_stop=t_stop
    )

    if not min_firing_rate:
        return np.corrcoef(st1_binned, st2_binned)[0, 1]

    ifr_1 = ifr(spiketrain_1, fs=fs, t_start=t_start, t_stop=t_stop).rename(
        columns={"ifr": "ifr_1"}
    )
    ifr_2 = ifr(spiketrain_2, fs=fs, t_start=t_start, t_stop=t_stop).rename(
        columns={"ifr": "ifr_2"}
    )
    df = pd.merge(ifr_1, ifr_2).assign(st1_binned=st1_binned, st2_binned=st2_binned)
    df["gmean"] = stats.gmean(df.loc[:, ["ifr_1", "ifr_2"]], axis=1)
    df = df.copy().loc[df["gmean"] > min_firing_rate]
    if len(df) == 0:
        return np.nan
    return np.corrcoef(df["st1_binned"], df["st2_binned"])[0, 1]
