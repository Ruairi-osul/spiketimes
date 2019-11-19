import numpy as np
import pandas as pd


def binned_spiketrain(spiketrain, fs, t_stop=None, t_start=None, as_df=False):
    if t_start is None:
        t_start = spiketrain[0]
    if t_stop is None:
        t_stop = spiketrain[-1]
    bins = np.arange(t_start, t_stop, 1 / fs)
    values, edges = np.histogram(spiketrain, bins=bins)
    df = pd.DataFrame({"time": edges[:-1], "spike_count": values})
    if as_df:
        return df
    else:
        return edges[1:], values
