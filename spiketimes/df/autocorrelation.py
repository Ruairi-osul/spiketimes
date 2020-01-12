import pandas as pd
import numpy as np
from itertools import combinations
from ..statistics import auto_corr, cross_corr
from .conversion import spikes_df_to_binned_df


def auto_corr_df(
    df: pd.core.frame.DataFrame,
    neuron_col: str = "neuron_id",
    spiketimes_col: str = "spiketimes",
    fs: int = 100,
    num_lags: int = 100,
):
    """
    Given a dataframe of spiketimes identified by a neuron column, 
    calculates the autocorrelation function for each neuron

    params:
        df: dataframe containing the data
        neuron_col: label of the column containing neuron ids
        fs: sampling rate to use to discretise the spiketimes
        num_lags: the number of time bins forwards and backwards to 
                  shift the correlation
    
    """
    return (
        df.groupby(neuron_col)
        .apply(
            lambda x: auto_corr(x[spiketimes_col], fs=fs, num_lags=num_lags, as_df=True)
        )
        .reset_index()
        .drop("level_1", axis=1)
    )


def cross_corr_df(
    df: pd.core.frame.DataFrame,
    neuron_col: str = "neuron_id",
    spiketimes_col: str = "spiketimes",
    fs: int = 100,
    num_lags: int = 100,
):
    """
    Given a dataframe of spiketimes identified by a neuron column, 
    calculates the cross correlation function between each pair of neurons

    params:
        df: dataframe containing the data
        neuron_col: label of the column containing neuron ids
        fs: sampling rate to use to discretise the spiketimes
        num_lags: the number of time bins forwards and backwards to 
                  shift the correlation
    
    """
    # multiprocessing
    neurons = df[neuron_col].unique()
    frames: list = []
    for neuron_1, neuron_2 in combinations(neurons, r=2):
        time_bins, cross_corr_ = cross_corr(
            df[df[neuron_col] == neuron_1][spiketimes_col],
            df[df[neuron_col] == neuron_1][spiketimes_col],
            fs=fs,
            num_lags=num_lags,
        )
        frames.append(
            pd.DataFrame(
                {
                    "neuron_1": neuron_1,
                    "neuron_2": neuron_2,
                    "time_bin": time_bins,
                    "cross_correlation": cross_corr_,
                }
            )
        )
    return pd.concat(frames, axis=0)
