import numpy as np
import pandas as pd


def binned_spiketrain(spiketrain, fs, t_stop=None, t_start=None, as_df=False):
    if isinstance(spiketrain, pd.core.series.Series):
        spiketrain = spiketrain.values
    if t_start is None:
        t_start = spiketrain[0]
    if t_stop is None:
        t_stop = spiketrain[-1]
    if not isinstance(spiketrain, np.ndarray):
        spiketrain = np.array(spiketrain)

    bins = np.arange(t_start, t_stop, 1 / fs)
    values, edges = np.histogram(spiketrain, bins=bins)

    if as_df:
        df = pd.DataFrame({"time": edges[:-1], "spike_count": values})
        return df
    else:
        return edges[1:], values


def bin_to_bool(binned_arr: np.ndarray):
    return np.where(binned_arr != 0, 1, 0)


def which_bin(spiketrain, bin_edges):
    idx = np.digitize(spiketrain, bin_edges) - 1
    bin_values = bin_edges[idx.tolist()]
    return idx, bin_values


def spike_count_around_event(spiketrain, binsize):
    pass
