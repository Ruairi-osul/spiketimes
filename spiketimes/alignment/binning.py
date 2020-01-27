import numpy as np
import pandas as pd


def binned_spiketrain(
    spiketrain: np.ndarray,
    fs: float,
    t_stop: float = None,
    t_start: float = None,
    as_df: bool = False,
):
    """
    Discretises a spiketrain. Creates bin edges at a specified sampling rate
    and returns spike counts per each bin. 
    Note: edges and counts have the same size

    params:
        spiketrain: np.ndarray of spiketimes in seconds
        fs: sampling rate used to discretise the spiketrain
        t_stop: right edge of final bin
        t_start: left edge of first bin
        as_df: if true, returns reseults as pd.DataFrame
    
    returns:
        edges: right edges of each bin
        counts: counts of spikes in each bin
         
    """
    if isinstance(spiketrain, pd.core.series.Series):
        spiketrain = spiketrain.values
    if t_start is None:
        t_start = spiketrain[0]
    if t_stop is None:
        t_stop = spiketrain[-1]
    if not isinstance(spiketrain, np.ndarray):
        spiketrain = np.array(spiketrain)

    bins = np.arange(t_start, t_stop + (1 / fs), 1 / fs)
    values, edges = np.histogram(spiketrain, bins=bins)

    if as_df:
        df = pd.DataFrame({"time": edges[:-1], "spike_count": values})
        return df
    else:
        return edges[1:], values


def binned_spiketrain_bins_provided(spiketrain: np.ndarray, bins: np.ndarray):
    """
    Given a numpy array of spiketimes and another of bin edges, 
    returns the counts of spikes in each bin 

    params:
        spiketrain: np.ndarray of spiketimes in seconds
        bins: np.ndarray of bin edges
    
    returns:
        np.ndarray of spike counts per bin
    """
    values, _ = np.histogram(spiketrain, bins=bins)
    return values


def bin_to_bool(binned_arr: np.ndarray):
    """
    Given an array of counts, returns an array of the same size
    with 1s in non-zero elements and 0s otherwise

    params:
        binned_arr: np.ndarray of counts
    
    returns:
        np.ndarray of 1s and 0s
    """
    return np.where(binned_arr != 0, 1, 0)


def which_bin(
    spiketrain: np.ndarray,
    bin_edges: np.ndarray,
    right: bool = True,
    as_df: bool = False,
):
    """`
    Given an array of spiketimes and an array of time bins (of the same units),
    returns the two numpy arrays of the same size as the spiketime array.
    The first array corresponds to the index of the encapsulating bin. The
    second the value of the corresponding bin. 
    
    params:
        spiketimes: np.ndarray of spiketimes in seconds
        bin_edges: edges of time bins
        right: if true, then values will be the right edge of the bin,
               left otherwise
    returns:
        np.ndarray bin indexes,
        np.ndarray of bin values
    """
    idx = np.digitize(spiketrain, bin_edges) - 1
    if right:
        idx_to_use = (idx + 1).tolist()
    else:
        idx_to_use = idx.tolist()
    bin_values = bin_edges[idx_to_use]

    if not as_df:
        return idx, bin_values
    else:
        return pd.DataFrame({"bin_idx": idx, "bin_values": bin_values})


def spike_count_around_event(
    spiketrain: np.ndarray, events: np.ndarray, binsize: float
):
    """
    Given an array of spiketimes and another of event times, calculates spike counts
    around the events.

    params:
        spiketrain: np.ndarray of spiketimes in seconds
        events: np.ndarray of event times in seconds
        binsize: size of timebin around event to count in seconds

    returns:
        np.ndarray of spike counts in seconds 
    """
    bins = np.repeat(events, 2).astype(np.float64)
    bins[1::2] += binsize
    spikecounts = binned_spiketrain_bins_provided(spiketrain, bins)
    return spikecounts[::2]
