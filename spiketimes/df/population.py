import pandas as pd
import numpy as np
from scipy import stats
from ..statistics import cross_corr


def population_coupling_df(
    df: pd.core.frame.DataFrame,
    neuron_col: str = "neuron_id",
    spiketimes_col: str = "spiketimes",
    bin_window: float = 0.01,
    num_lags: int = 100,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Given a pandas.DataFrame containing spikeing data of simultaneous spiking neurons,
    for each neuron, calculates a metric of population spike coupling. The metric is 
    calculated by calculating and zscore standardising cross correlation between an individual
    spike train and the merged spiketrain of all other neurons. Large Z score cross correlation
    at lag=0 is indicative of high population coupling.

    params:
        df: pandas.DataFrame containing the spikeing data
        neuron_col: label of column containing neuron identifiers
        spiketimes_col: label of column containing spiketimes in seconds
        bin_window: size of time bin used in cross correlation in seconds
        num_lags: the number of labs to calculate cross correlation around 0
        t_start: the time after cross correlation will be calculated
        t_stop: the time before which cross correlation will be calculated

    returns:
        a pandas.DataFrame containing columns labelled {neuron_col, "time_sec", "zscore"}  
    """
    ROUNDING_PRECISION = 5
    frames: list = []
    neurons = df[neuron_col].unique()
    for neuron in neurons:
        neuron_spiketrain = df[df[neuron_col] == neuron][spiketimes_col].values
        population_spiketrain = np.sort(
            df[df[neuron_col] != neuron][spiketimes_col].values
        )
        t, cc = cross_corr(
            spiketrain_1=neuron_spiketrain,
            spiketrain_2=population_spiketrain,
            bin_window=bin_window,
            num_lags=num_lags,
            as_df=False,
            t_start=t_start,
            t_stop=t_stop,
            delete_0_lag=False,
        )
        z = stats.zscore(cc)
        t = np.round(t, ROUNDING_PRECISION)
        df_out = pd.DataFrame({"time_sec": t, "zscore": z, neuron_col: neuron})
        frames.append(df_out)
    df = pd.concat(frames, axis=0)
    return df


def population_coupling_df_by(
    df: pd.core.frame.DataFrame,
    neuron_col: str = "neuron_id",
    spiketimes_col: str = "spiketimes",
    by_col: str = "session_name",
    bin_window: float = 0.01,
    num_lags: int = 100,
    t_start: float = None,
    t_stop: float = None,
):
    """    
    Given a pandas.DataFrame containing spikeing data of multiple neurons, grouped by some variable,
    within each group, for each neuron, calculates a metric of population spike coupling. 
    The metric is  calculated by calculating and zscore standardising cross correlation 
    between an individual spike train and the merged spiketrain of all other neurons. 
    Large Z score cross correlation at lag=0 is indicative of high population coupling.

    params:
        df: pandas.DataFrame containing the spikeing data
        neuron_col: label of column containing neuron identifiers
        spiketimes_col: label of column containing spiketimes in seconds
        by_col: label of column indicating group member ship. All neurons within one group
                should be simultaneously recorded for the metric to be meaningful.
        bin_window: size of time bin used in cross correlation in seconds
        num_lags: the number of labs to calculate cross correlation around 0
        t_start: the time after cross correlation will be calculated
        t_stop: the time before which cross correlation will be calculated

    returns:
        a pandas.DataFrame containing columns labelled {neuron_col, "time_sec", "zscore"}  
    """
    return (
        df.groupby(by_col)
        .apply(
            lambda x: population_coupling_df(
                x,
                neuron_col=neuron_col,
                spiketimes_col=spiketimes_col,
                bin_window=bin_window,
                num_lags=num_lags,
                t_start=t_start,
                t_stop=t_stop,
            )
        )
        .reset_index()
        .drop("level_1", axis=1)
    )

