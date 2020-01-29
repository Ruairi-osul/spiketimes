import pandas as pd
import numpy as np
from ..statistics import cv_isi, cv2_isi


def mean_firing_rate_ifr_by_neuron(
    df: pd.core.frame.DataFrame, neuron_col: str = "neuron_id", ifr_col: str = "ifr"
):
    """
    Given a dataframe of intantaneous firing rates of many neurons, calculates the 
    mean firing rate of each neuron

    params:
        df: the df containg the data
        neuron_col: label for the column containing neuron ids
        ifr_col: label for the column containing the ifr values

    returns:
        a pd.DataFrame containing with columns {neuron_id, mean_firing_rate} 
    """

    return (
        df.groupby(neuron_col)
        .apply(lambda x: mean_firing_rate_ifr(x, ifr_col=ifr_col))
        .reset_index()
        .rename(columns={0: "mean_firing_rate"})
    )


def mean_firing_rate_ifr(df: pd.core.frame.DataFrame, ifr_col: str = "ifr"):
    """
    Given a dataframe containing a instantaneous firing rate column,
    calculates the spiketrain's mean firing rate

    params:
        df: dataframe containing the data
        ifr_col: column containing  instantaneous firing rate estimates
    returns:
        a scaler of mean firing rate
    """
    return df[ifr_col].mean()


def cv_isi_by_neuron(
    df: pd.core.frame.DataFrame,
    spiketimes_col: str = "spiketimes",
    neuron_col: str = "neuron_id",
):
    """
    Given a dataframe containing spiketimes indexed by neuron,
    calculates the coefficient of variation of inter-spike-intervals
    for each neuron. CV_ISI can be used as a metric of regularity of spiking.

    params:
        df: the dataframe containing the data
        spiketimes_col: label of column containing spiketimes
        neuron_col: label of column indexes the neuron responsible 
                    for the spike
    returns:
        a dataframe with columns neuron_col and cv_isi
    """
    return (
        df.groupby(neuron_col)
        .apply(lambda x: cv_isi(x[spiketimes_col].values))
        .reset_index()
        .rename(columns={0: "cv_isi"})
    )


def cv2_isi_by_neuron(
    df: pd.core.frame.DataFrame,
    spiketimes_col: str = "spiketimes",
    neuron_col: str = "neuron_id",
):
    """
    Given a dataframe containing spiketimes indexed by neuron,
    calculates the cv2 of interspike intervals for each neuron. 
    CV2_ISI can be used as a metric of regularity of spiking.

    params:
        df: the dataframe containing the data
        spiketimes_col: label of column containing spiketimes
        neuron_col: label of column indexes the neuron responsible 
                    for the spike
    returns:
        a dataframe with columns neuron_col and cv2_isi
    """
    return (
        df.groupby(neuron_col)
        .apply(lambda x: cv2_isi(x[spiketimes_col].values))
        .reset_index()
        .rename(columns={0: "cv2_isi"})
    )


def fraction_silent_by_neuron(
    df: pd.core.frame.DataFrame,
    bool_col: str = "has_spike",
    neuron_col: str = "neuron_id",
):
    """
    Given a df containing a column identifying neurons and a boolean column
    refering to whether a spike occured, calculates the fraction of bins containing
    a spike

    params:
        df: df containing the data
        bool_col: 

    returns:
        a pandas dataframe containing columns {neuron_col, fraction_silent} 
    """
    return (
        df.groupby(neuron_col)
        .apply(lambda x: np.mean(x[bool_col]))
        .reset_index()
        .rename(columns={0: "fraction_silent"})
    )

