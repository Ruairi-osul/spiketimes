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
    Get event counts at regular time bins.

    Args:
        spiketrain: A numpy array of spiketimes in seconds
        fs: The sampling rate used to create time bins. The number of samples per second.
        t_start: The left edge of first bin
        t_stop: The right edge of final bin
        as_df: If true, returns reseults as a pandas DataFrame
    Returns:
        edges, counts
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
    Get event counts at specified time bins.

    Args:
        spiketrain: A numpy array of spiketimes in seconds
        bins: A numpy array of bin edges
    Returns:
        A numpy array of spike counts per bin
    """
    values, _ = np.histogram(spiketrain, bins=bins)
    return values


def bin_to_bool(binned_arr: np.ndarray):
    """
    Convert a binned array to a binary array: 1s for nonzero elements, 0s for zeros.

    Args:
        binned_arr: A numpy array of counts
    Returns:
        A numpy array of 1s and 0s
    """
    return np.where(binned_arr != 0, 1, 0)


def which_bin(
    spiketrain: np.ndarray,
    bin_edges: np.ndarray,
    before: float = None,
    allow_before: bool = False,
    max_latency: float = None,
    as_df: bool = False,
):
    """
    Get the corresponding bin for each spike in a spiketrain.

    Useful for splitting spiketrains into trials.

    Args:
        spiketrain: A numpy array of spiketimes
        bin_edges: A numpy array of bin edges
        before: The time window before each event to include in the alignment.
        allow_before: If False, spikes occuring before the first time are returned as NaN
        max_latency: If specified, spikes occuring this quantity after the maximum bin_edge
                     are returned as NaN
        as_df: Whether to returned the result as a pandas DataFrame.
    Returns:
        bin_idx, bin_values
    """
    if before is not None:
        idx = np.digitize(spiketrain, (bin_edges - before)) - 1
    else:
        idx = np.digitize(spiketrain, bin_edges) - 1

    idx_to_use = idx.tolist()
    bin_values = (bin_edges[idx_to_use]).astype(np.float)
    if not allow_before:
        nan_mask = idx < 0
        idx = idx.astype(np.float)
        bin_values[nan_mask] = np.nan
        idx[nan_mask] = np.nan
    if max_latency is not None:
        latency_to_max = spiketrain - np.max(bin_edges)
        idx[latency_to_max > max_latency] = np.nan
        bin_values[latency_to_max > max_latency] = np.nan
    if not as_df:
        return idx, bin_values
    else:
        return pd.DataFrame({"bin_idx": idx, "bin_values": bin_values})


def spike_count_around_event(
    spiketrain: np.ndarray, events: np.ndarray, binsize: float
):
    """
    Calculate the spike count around events.

    Args:
        spiketrain: A numpy array of spiketimes
        events: A numpy array of events
        binsize: The size of window in which to count spikes
    Returns:
        A numpy array of spike counts per each event
    """
    bins = np.repeat(events, 2).astype(np.float64)
    bins[1::2] += binsize
    spikecounts = binned_spiketrain_bins_provided(spiketrain, bins)
    return spikecounts[::2]
