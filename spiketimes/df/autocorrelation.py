import pandas as pd
import numpy as np
from itertools import combinations
import multiprocessing
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

    neuron_combs = list(combinations(neurons, r=2))
    args = [
        [
            df[df[neuron_col] == neuron_1][spiketimes_col].values,
            df[df[neuron_col] == neuron_2][spiketimes_col].values,
            fs,
            num_lags,
            True,
        ]
        for neuron_1, neuron_2 in neuron_combs
    ]

    with multiprocessing.Pool() as p:
        res = p.starmap(cross_corr, args)

    for neuron_comb, r in zip(neuron_combs, res):
        r["neuron_1"] = neuron_comb[0]
        r["neuron_2"] = neuron_comb[1]
    return pd.concat(res, axis=0)

    # frames: list = []
    # for neuron_1, neuron_2 in combinations(neurons, r=2):
    #     time_bins, cross_corr_ = cross_corr(
    #         df[df[neuron_col] == neuron_1][spiketimes_col],
    #         df[df[neuron_col] == neuron_1][spiketimes_col],
    #         fs=fs,
    #         num_lags=num_lags,
    #     )
    #     frames.append(
    #         pd.DataFrame(
    #             {
    #                 "neuron_1": neuron_1,
    #                 "neuron_2": neuron_2,
    #                 "time_bin": time_bins,
    #                 "cross_correlation": cross_corr_,
    #             }
    #         )
    #     )
    # return pd.concat(frames, axis=0)
