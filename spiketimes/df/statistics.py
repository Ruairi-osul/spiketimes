import pandas as pd
import numpy as np
import spiketimes
import spiketimes.statistics
from .binning import binned_spiketrain


def mean_firing_rate_by(
    df: pd.core.frame.DataFrame,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    t_start: float = None,
    t_stop: float = None,
):
    """
    Estimate the mean firing rate of each spiketrain.

    Firing rate caluclated by summing spikes and dividing by total time.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column identifying the spiketrain responsible for the spike
        t_start: Time point at which to start. Defaults to time of first spike in df.
        t_stop: Maximum timepoint. Defaults to last spike in df.
    Returns:
        A DataFrame containing mean firing rate by neuron
    """
    if t_start is None:
        t_start = df[spiketimes_col].min()
    if not t_stop:
        t_stop = df[spiketimes_col].max()
    return (
        df.groupby(spiketrain_col)
        .apply(
            lambda x: spiketimes.statistics.mean_firing_rate(
                x[spiketimes_col].values, t_start=t_start, t_stop=t_stop,
            )
        )
        .reset_index()
        .rename(columns={0: "mean_firing_rate"})
    )


def ifr_by(
    df: pd.core.frame.DataFrame,
    fs: float = 1,
    sigma: float = None,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    t_start: float = None,
    t_stop: float = None,
):
    """
    Estimate firing rate for each spiketrain at a regular sampling rate.

    Args:
        df: A pandas DataFrame containing the spikes data
        fs: The sampling rate at which to estimate firing rate
        sigma: Hypterparameter controlling smoothing for firing rate estimates
        spiketimes_col: The label of the column in df containing spiketimes
        spiketrain_col: The label of the column in df containing spiketrain idendifiers
                        (which spiketrain was responsible for the spike)
        t_start: Time point at which to start firing rate estimates. Defaults to time of first spike in df.
        t_stop: Time point of maximum firing rate estimate. Defaults to last spike in df.
    Returns:
        A pandas DataFrame with one row per timepoint per spiketrain with column `ifr` identifying
        firing rate estimates.
    """
    if t_start is None:
        t_start = df[spiketimes_col].min()
    if not t_stop:
        t_stop = df[spiketimes_col].max()
    return (
        df.groupby(spiketrain_col)
        .apply(
            lambda x: spiketimes.statistics.ifr(
                x[spiketimes_col].values,
                fs=fs,
                sigma=sigma,
                t_start=t_start,
                t_stop=t_stop,
                as_df=True,
            )
        )
        .reset_index()
        .drop("level_1", axis=1)
    )


def mean_firing_rate_ifr_by(
    df: pd.core.frame.DataFrame,
    fs: float = 1,
    sigma: float = None,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    exclude_below: float = None,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Estimate mean firing rate of each neuron by first estimating firing rate at a regular interval
    and then taking the median.

    Args:
        df: A pandas Dataframe containing the spike data
        fs: The sampling rate at which to estimate firing rate
        sigma: Parameter contolling smoothing level of firing rate estiamtes.
        exclude_below: If specified, firing rates below this value will not be included in the median calculation.
        spiketimes_col: The label of the column containing the spiketimes
        spiketrain_col: The label of the column in df containing spiketrain idendifiers
                        (which spiketrain was responsible for the spike)
        t_start: Time point at which to start firing rate estimates. Defaults to time of first spike in df.
        t_stop: Time point of maximum firing rate estimate. Defaults to last spike in df.
    Returns:
        A pandas DataFrame containing one row per spiketrain as well as its firing rate estimate.
    """
    if t_start is None:
        t_start = df[spiketimes_col].min()
    if not t_stop:
        t_stop = df[spiketimes_col].max()
    return (
        df.groupby(spiketrain_col)
        .apply(
            lambda x: spiketimes.statistics.mean_firing_rate_ifr(
                x[spiketimes_col].values,
                fs=fs,
                sigma=sigma,
                exclude_below=exclude_below,
            )
        )
        .reset_index()
        .rename(columns={0: "mean_firing_rate_ifr"})
    )


def cv_isi_by(
    df: pd.core.frame.DataFrame,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
):
    """
    Calculate the coefficient of variation of interspike intervals for each spiketrain in a DataFrame.

    The cv_isi is a metric of spike regularity. Values near 1 are typical of poisson processes. Values near 0
    indicate very regular processes.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column identifying the spiketrain responsible for the spike
    Returns:
        A DataFrame containing cv_isi by neuron
    """
    return (
        df.groupby(spiketrain_col)
        .apply(lambda x: spiketimes.statistics.cv_isi(x[spiketimes_col].values))
        .reset_index()
        .rename(columns={0: "cv_isi"})
    )


def cv2_isi_by(
    df: pd.core.frame.DataFrame,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
):
    """
    Calculate cv2 of interspike intervals of each spiketrain.

    cv2 is a metric related to the coefficient of variation. It is adapted to be suitable long-period spiketrains.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column identifying the spiketrain responsible for the spike
    Returns:
        A DataFrame containing cv2_isi by neuron
    """
    return (
        df.groupby(spiketrain_col)
        .apply(lambda x: spiketimes.statistics.cv2_isi(x[spiketimes_col].values))
        .reset_index()
        .rename(columns={0: "cv2_isi"})
    )


def fraction_silent_by(
    df: pd.core.frame.DataFrame,
    binsize: float = 1,
    silent_threshold: float = 0.5,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    t_start: float = None,
    t_stop: float = None,
):
    """
    Estimate the fraction of time a spiketrain was inactivate.

    Estimate calculated by binning spikes into time bins and calculating the proportion of spikes falling below
    a specified threshold.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        binsize: The time period in seconds to use when binning spikes.
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column identifying the spiketrain responsible for the spike
        t_start: Time point at which to start. Defaults to time of first spike in df.
        t_stop: Maximum timepoint. Defaults to last spike in df.
    Returns:
        A pandas DataFrame containing fraction silent estimates by neuron.
    """
    if t_start is None:
        t_start = df[spiketimes_col].min()
    if not t_stop:
        t_stop = df[spiketimes_col].max()
    fs = 1 / binsize
    return (
        binned_spiketrain(
            df,
            spiketimes_col=spiketimes_col,
            spiketrain_col=spiketrain_col,
            fs=fs,
            t_start=t_start,
            t_stop=t_stop,
        )
        .groupby(spiketrain_col)
        .apply(lambda x: np.mean(x["spike_count"] > silent_threshold))
        .reset_index()
        .rename(columns={0: "fraction_silent"})
    )


def auc_roc_test_by(
    df: pd.core.frame.DataFrame,
    n_boot: int = 1000,
    return_distance_from_chance: bool = False,
    spikecount_col: str = "spike_count",
    spiketrain_col: str = "spiketrain",
    condition_col: str = "cond",
):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve of spike counts for each spiketrain.

    The AUCROC can be used as a metric of the separability of two distrobutions. Each spiketrain must have been recorded
    in both conditions during multiple trials. Significance tested using a permutation test.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        n_boot: The number of permutation replicates to draw.
        spikecount_col: The label of the column containing spikecounts
        spiketrain_col: The label of the column identifying the spiketrain responsible for the spike
        condition_col: A categorical column containing 0 for the baseline condition and 1 for the experimental condition
        return_distance_from_chance: If True, returns distance from 0.5
    Returns:
        A pandas DataFrame containing one row per spiketrain with columns {'spiketrain', 'AUCROC', 'p'}
    """
    return (
        df.groupby(spiketrain_col)
        .apply(
            lambda x: pd.Series(
                spiketimes.statistics.auc_roc_test(
                    x[spikecount_col].values,
                    x[condition_col].values,
                    n_boot=n_boot,
                    return_distance_from_chance=return_distance_from_chance,
                )
            )
        )
        .reset_index()
        .rename(columns={0: "AUCROC", 1: "p"})
    )


def diffmeans_test_by(
    df: pd.core.frame.DataFrame,
    n_boot: int = 1000,
    return_distance_from_chance: bool = False,
    spikecount_col: str = "spike_count",
    spiketrain_col: str = "spiketrain",
    condition_col: str = "cond",
):
    """
    Calculates the difference between means of spike counts for each spike in a data frame and also tests
    significance using a permutation test.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        n_boot: The number of permutation replicates to draw.
        spikecount_col: The label of the column containing spikecounts
        spiketrain_col: The label of the column identifying the spiketrain responsible for the spike
        condition_col: A categorical column containing 0 for the baseline condition and 1 for the experimental condition
    Returns:
        A pandas DataFrame containing one row per spiketrain with columns {'spiketrain', 'diff_of_means', 'p'}
    """
    return (
        df.groupby(spiketrain_col)
        .apply(
            lambda x: pd.Series(
                spiketimes.statistics.diffmeans_test(
                    x[spikecount_col].values, x[condition_col].values, n_boot=n_boot,
                )
            )
        )
        .reset_index()
        .rename(columns={0: "diff_of_means", 1: "p"})
    )
