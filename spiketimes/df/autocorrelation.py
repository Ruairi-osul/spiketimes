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
    Given a dataframe containing spiketimes where each spike is indexed 
    by neuron, calculates autocorrealtion for each neuron and aggregates the
    results.

    params:
        df: dataframe containing spiketimes, neuron ids, group ids,
        neuron_col: label of column containing neuron ids
        spiketimes_col: label of column containing spiketimes
        tail: which tail to use for hypothesis tests ('two tail recommended')
        bin_window: bin width for cross correlation
        num_lags: number of lags forward and backwards around lag 0 to return 
    returns:
        pandas DataFrame with columns neuron_1, neuron_2, autocorrelation
    
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
    Given a dataframe containing spiketimes where each spike is indexed by neuron,
    calculates crosscorrelation between each pair of neurons. 

    params:
        df: dataframe containing spiketimes, neuron ids, group ids,
        neuron_col: label of column containing neuron ids
        spiketimes_col: label of column containing spiketimes
        tail: which tail to use for hypothesis tests ('two tail recommended')
        bin_window: bin width for cross correlation
        num_lags: number of lags forward and backwards around lag 0 to return 
        t_start: if specified, no spikes before this limit will be considered
        t_stop: if specified, no spikes before after limit will be considered
    returns:
        pandas DataFrame with columns neuron_1, neuron_2, crosscorrelation
    
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
    Given a dataframe containing spiketimes where each spike is indexed by neuron,
    calculates crosscorrelation and between each pair of neurons. Also calculates 
    significance by assuming the cross correlation bin values follow a 
    poisson distrobution.

    params:
        df: dataframe containing spiketimes, neuron ids, group ids,
        neuron_col: label of column containing neuron ids
        spiketimes_col: label of column containing spiketimes
        tail: which tail to use for hypothesis tests ('two tail recommended')
        bin_window: bin width for cross correlation
        num_lags: number of lags forward and backwards around lag 0 to return 
        t_start: if specified, no spikes before this limit will be considered
        t_stop: if specified, no spikes before after limit will be considered
        adjust_p: whether to adjust p values for multiple comparisons (strongly recommended)
        p_adjust_method: method to use for p adjustment {'Benjamini-Hochberg', 
                                                        'Bonferroni', 'Bonferroni-Holm'}
        n_cores: number of cores to use for multiprocessing
    returns:
        pandas DataFrame with columns neuron_1, neuron_2, crosscorrelation, p
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
    """
    Given a dataframe containing spiketimes where each spike is indexed by neuron
    and group, calculates cross correlation between each pair of neurons in different
    groups. Also calculates significance by assuming the cross correlation bin 
    values follow a poisson distrobution.

    params:
        df: dataframe containing spiketimes, neuron ids, group ids,
        neuron_col: label of column containing neuron ids
        spiketimes_col: label of column containing spiketimes
        group_col: label of column containing group ids
        tail: which tail to use for hypothesis tests ('two tail recommended')
        bin_window: bin width for cross correlation
        num_lags: number of lags forward and backwards around lag 0 to return 
        t_start: if specified, no spikes before this limit will be considered
        t_stop: if specified, no spikes before after limit will be considered
        adjust_p: whether to adjust p values for multiple comparisons (strongly recommended)
        p_adjust_method: method to use for p adjustment {'Benjamini-Hochberg', 
                                                         'Bonferroni', 'Bonferroni-Holm'}
        n_cores: number of cores to use for multiprocessing
    returns:
        pandas DataFrame with columns neuron_1, neuron_2, group_1, group_2, 
                                      crosscorrelation, p
    """
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

