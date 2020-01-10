import numpy as np
import pandas as pd
from scipy import stats
from ..alignment import binned_spiketrain
from ..statistics import ifr
from ..simulate import shuffled_isi_spiketrains


def spike_count_correlation_test(
    spiketrain_1: np.ndarray,
    spiketrain_2: np.ndarray,
    fs: int,
    n_boot: int = 500,
    min_firing_rate: float = None,
    t_start: float = None,
    t_stop: float = None,
):
    n_surrogates = int(n_boot * 2)
    st1_shuffled = shuffled_isi_spiketrains(spiketrain_1, n=n_surrogates)
    st2_shuffled = shuffled_isi_spiketrains(spiketrain_2, n=n_surrogates)
    pass


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
