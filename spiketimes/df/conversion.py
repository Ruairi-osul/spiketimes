import numpy as np
import pandas as pd
from ..statistics import ifr
from ..alignment import binned_spiketrain, bin_to_bool


def list_to_df(spiketrains: list, indexes: list = None):
    """
    convert a list of spiketrains into a tidy dataframe of spiketimes

    Args:
        spiketrains: list of spiketrains
        indexes: optional list of labels for the of the spiketrains

    Returns:
        a pandas DataFrame containing one spike and id label per row
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


def spikes_df_to_binned_df(
    df: pd.core.frame.DataFrame,
    spiketimes_col: str = "spiketimes",
    neuron_col: str = "neuron_id",
    fs: str = 1,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Given a df containing a column of spiketimes, convert to a binned df
    with a given sampling interval

    Args:
        df: pandas dataframe to be converted
        spiketimes_col: label of the column containing the spike data.
                        data should be in unit of seconds
        fs: desired sampling frequency in seconds
        t_start: the time after which the first bin will start. defaults to 0.
        t_stop: the maximum time for the time bins 

    Returns:
        a pandas DataFrame containing the binned data
    """
    if t_stop is None:
        t_stop = df[spiketimes_col].values[-1]

    return (
        df.groupby(neuron_col)
        .apply(
            lambda x: binned_spiketrain(
                x[spiketimes_col], fs=fs, t_start=t_start, t_stop=t_stop, as_df=True
            )
        )
        .reset_index()
        .drop("level_1", axis=1)
    )


def df_binned_to_bool(binned_ser: pd.core.series.Series):
    """
    Convert a binned Series to a boolean series of 1s and 0s.
    """
    return bin_to_bool(binned_ser)


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

