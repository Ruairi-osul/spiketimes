import numpy as np
import pandas as pd
from ..statistics import ifr


def list_to_df(spiketrains: list, indexes=None):
    """
    convert a list of spiketrains into a tidy dataframe of spiketimes
    params:
        spiketrains: list of spiketrains
        indexes: optional list of 
    """
    if indexes is None:
        indexes = np.arange(len(spiketrains))
    else:
        assert len(spiketrains) == len(
            indexes
        ), "index and spiketrains must be the same size"

    df_list = [
        pd.DataFrame({"spiketrain": index, "timepoint_s": spiketrain})
        for index, spiketrain in zip(indexes, spiketrains)
    ]

    return pd.concat(df_list)


def df_ifr(df, spiketime_col="spike_time_samples", fs=1, t_start=None, t_stop=None):
    "Given a spiketime df of a single neuron, returns an estimate of instantaneous firing rate"
    if t_stop is None:
        t_stop = df[spiketime_col].max()
    if t_start is None:
        t_start = df[spiketime_col].min()
    return df.sort_values(spiketime_col, axis="rows").pipe(
        lambda x: ifr(x[spiketime_col], fs=fs, t_start=t_start, t_stop=t_stop)
    )


def ifr_by_neuron(
    df,
    neuron_col,
    spiketime_col="spike_time_samples",
    ifr_fs=1,
    t_start=None,
    t_stop=None,
):
    """Converts a df of spiketimes to a df of instantaneous firing rate.
    Groups by neuron_col and applies df_ifr to each"""
    if t_stop is None:
        t_stop = df[spiketime_col].max()
    if t_start is None:
        t_start = df[spiketime_col].min()
    return (
        df.groupby(neuron_col)
        .apply(
            lambda x: df_ifr(
                x,
                spiketime_col=spiketime_col,
                fs=ifr_fs,
                t_start=t_start,
                t_stop=t_stop,
            )
        )
        .reset_index()
        .drop("level_1", axis=1)
    )

