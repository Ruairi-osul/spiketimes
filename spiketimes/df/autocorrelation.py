import pandas as pd
import numpy as np
from itertools import combinations, product
import multiprocessing
from ..statistics import auto_corr, cross_corr, cross_corr_test
from .conversion import spikes_df_to_binned_df
from ..statistics.utils import p_adjust


def auto_corr_df(
    df: pd.core.frame.DataFrame,
    neuron_col: str = "neuron_id",
    spiketimes_col: str = "spiketimes",
    bin_window: int = 0.01,
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
            lambda x: auto_corr(
                x[spiketimes_col], bin_window=bin_window, num_lags=num_lags, as_df=True
            )
        )
        .reset_index()
        .drop("level_1", axis=1)
    )


def cross_corr_df(
    df: pd.core.frame.DataFrame,
    neuron_col: str = "neuron_id",
    spiketimes_col: str = "spiketimes",
    bin_window: float = 0.01,
    num_lags: int = 100,
    t_start: float = None,
    t_stop: float = None,
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
    neurons = df[neuron_col].unique()

    neuron_combs = list(combinations(neurons, r=2))
    args = [
        [
            df[df[neuron_col] == neuron_1][spiketimes_col].values,
            df[df[neuron_col] == neuron_2][spiketimes_col].values,
            bin_window,
            num_lags,
            True,
            t_start,
            t_stop,
        ]
        for neuron_1, neuron_2 in neuron_combs
    ]

    with multiprocessing.Pool() as p:
        res = p.starmap(cross_corr, args)

    for neuron_comb, r in zip(neuron_combs, res):
        r["neuron_1"] = neuron_comb[0]
        r["neuron_2"] = neuron_comb[1]
    return pd.concat(res, axis=0)


def cross_corr_df_test(
    df: pd.core.frame.DataFrame,
    neuron_col: str = "neuron_id",
    spiketimes_col: str = "spiketimes",
    tail: str = "two_tailed",
    bin_window: float = 0.01,
    num_lags: int = 100,
    t_start: int = None,
    t_stop: int = None,
    adjust_p: bool = True,
    p_adjust_method: str = "Benjamini-Hochberg",
    n_cores: int = None,
):
    """
    Given a dataframe of spiketimes identified by a neuron column, 
    calculates the cross correlation function between each pair of neurons.
    Also test significance of each time lag using a bootstrap approach inwhich
    crosscorrelation at each lag is compared to cross correlation of jitter
    surrogate spiketrains. 

    params:
        df: dataframe containing the data
        neuron_col: label of the column containing neuron ids
        bin_window: size of bins in seconds used to discretise the spiketrain
        num_lags: the number of time bins forwards and backwards to 
                  shift the correlation
    """
    neurons = df[neuron_col].unique()
    neuron_combs = list(combinations(neurons, r=2))
    args = [
        [
            df[df[neuron_col] == neuron_1][spiketimes_col].values,
            df[df[neuron_col] == neuron_2][spiketimes_col].values,
            bin_window,
            num_lags,
            True,
            t_start,
            t_stop,
            tail,
            False,
            None,
        ]
        for neuron_1, neuron_2 in neuron_combs
    ]

    if n_cores:
        with multiprocessing.Pool(n_cores) as p:
            res = p.starmap(cross_corr_test, args)
    else:
        with multiprocessing.Pool() as p:
            res = p.starmap(cross_corr_test, args)

    for neuron_comb, r in zip(neuron_combs, res):
        r["neuron_1"] = neuron_comb[0]
        r["neuron_2"] = neuron_comb[1]
    df = pd.concat(res, axis=0)

    if adjust_p:
        df["p"] = p_adjust(df["p"].values, method=p_adjust_method)

    return df


def cross_corr_between_groups_test(
    df: pd.core.frame.DataFrame,
    neuron_col: str = "neuron_id",
    spiketimes_col: str = "spiketimes",
    group_col: str = "group",
    tail: str = "two_tailed",
    bin_window: float = 0.01,
    num_lags: int = 100,
    t_start: int = None,
    t_stop: int = None,
    adjust_p: bool = True,
    p_adjust_method: str = "Benjamini-Hochberg",
    n_cores: int = None,
):
    frames: list = []
    groups = df[group_col].unique()
    for group_1, group_2 in combinations(groups, r=2):
        neurons_group_1 = df.loc[df[group_col] == group_1][neuron_col].unique()
        neuron_group_2 = df.loc[df[group_col] == group_2][neuron_col].unique()
        neuron_combs = list(product(neurons_group_1, neuron_group_2))
        args = [
            [
                df[df[neuron_col] == neuron_1][spiketimes_col].values,
                df[df[neuron_col] == neuron_2][spiketimes_col].values,
                bin_window,
                num_lags,
                True,
                t_start,
                t_stop,
                tail,
                False,
                None,
            ]
            for neuron_1, neuron_2 in neuron_combs
        ]
        if n_cores:
            with multiprocessing.Pool(n_cores) as p:
                res = p.starmap(cross_corr_test, args)
        else:
            with multiprocessing.Pool() as p:
                res = p.starmap(cross_corr_test, args)

        for neuron_comb, r in zip(neuron_combs, res):
            r["neuron_1"] = neuron_comb[0]
            r["neuron_2"] = neuron_comb[1]

        dfg = pd.concat(res, axis=0)
        dfg["group_1"] = group_1
        dfg["group_2"] = group_2
        frames.append(dfg)

    df = pd.concat(frames, axis=0)
    if adjust_p:
        df["p"] = p_adjust(df["p"].values, method=p_adjust_method)
    return df

