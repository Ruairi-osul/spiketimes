import pandas as pd
from ..surrogates import shuffled_isi_spiketrains as shuffled_isi_spiketrains_arr
from ..surrogates import jitter_spiketrains as jitter_spiketrains_arr
from .conversion import list_to_df


def shuffled_isi_spiketrains(
    df: pd.core.frame.DataFrame,
    spiketimes_col: str = "spiketimes",
    n: int = 1,
    returned_surrogate_label: str = "surrogate",
):
    """
    Create multiple shuffled-ISI surrogates from a single parent spiketrain.

    Given a dataframe containing a spiketimes from a neuron, returns a dataframe
    of n surrogate spiketrians by shuffling inter-spike-intervals

    Args:
        df: dataframe containing the data
        spiketimes_col: label of column containing spiketimes
        n: number of surrogate spiketrains to replicate
        returned_surrogate_label: column label indicating surrogate identity
    Returns:
        a pandas dataframe of spiketimes indexed by spiketrain
    """
    return list_to_df(
        shuffled_isi_spiketrains_arr(df[spiketimes_col].values, n=n),
        returned_spiketrain_label=returned_surrogate_label,
    )


def jitter_spiketrains(
    df: pd.core.frame.DataFrame,
    jitter_window_size: float,
    spiketimes_col: str = "spiketimes",
    n: int = 1,
    returned_surrogate_label: str = "surrogate",
):
    """
    Create multiple spiketime-jittered surrogates from a single parent spiketrain.

    Given a dataframe containing a spiketimes from a neuron, returns a dataframe
    of n surrogate spiketrians by binning spike counts and randomly dispersing
    spiketimes within each timebin.

    Args:
        df: dataframe containing the data
        spiketimes_col: label of column containing spiketimes
        jitter_window_size: binwidth in seconds used to bin spike counts
        n: number of surrogate spiketrains to replicate
        returned_surrogate_label: column label indicating surrogate identity
    Returns:
        a pandas dataframe of spiketimes indexed by spiketrain
    """
    return list_to_df(
        jitter_spiketrains_arr(
            df[spiketimes_col].values, jitter_window_size=jitter_window_size, n=n
        ),
        returned_spiketrain_label=returned_surrogate_label,
    )


def shuffled_isi_spiketrains_by(
    df: pd.core.frame.DataFrame,
    spiketimes_col: str = "spiketimes",
    by_col: str = "spiketrain",
    n: int = 1,
):
    """
    Craete multiple shuffled-ISI surrogates for spiketrain in a dataframe.

    Given a dataframe of spiketimes grouped by spiketrain another column,
    generates n surrogates from each spiketrain. Surrgates are generated by shuffling
    inter-spike-intervals.

    Args:
        df: dataframe containing the data
        spiketimes_col: label of column containing spiketimes
        by_col: label of column indicating group (e.g. neuron_id, spiketrain_id or trial_id)
        n: number of surrogates to generate per group
    Returns:
        A pandas dataframe containing surrogate spiketrain indexed by a 'surrogate_replicate' column
    """
    return (
        df.groupby(by_col)
        .apply(
            lambda x: shuffled_isi_spiketrains(x, spiketimes_col=spiketimes_col, n=n)
        )
        .reset_index()
        .drop("level_1", axis=1)
    )


def jitter_spiketrains_by(
    df: pd.core.frame.DataFrame,
    jitter_window_size: float,
    spiketimes_col: str = "spiketimes",
    by_col: str = "spiketrain",
    n: int = 1,
):
    """
    Craete multiple shuffled-ISI surrogates for spiketrain in a dataframe.

    Given a dataframe of spiketimes grouped by spiketrain another column,
    generates n surrogates from each spiketrain. Surrogates generated by generating spikecounts
    from the parent from the start point of the spiketrain untill the end, then
    generating surrogates with the same number of spikes in each time bin, but with spiketimes
    randomised.

    Args:
        df: dataframe containing the data
        jitter_window_size: binwidth in seconds used to bin spike counts
        spiketimes_col: label of column containing spiketimes
        by_col: label of column indicating group (e.g. neuron_id, spiketrain_id or trial_id)
        n: number of surrogates to generate per group
    Returns:
        A pandas dataframe containing surrogate spiketrain indexed by a 'surrogate_replicate' column
    """
    return (
        df.groupby(by_col)
        .apply(
            lambda x: jitter_spiketrains(
                x,
                spiketimes_col=spiketimes_col,
                n=n,
                jitter_window_size=jitter_window_size,
            )
        )
        .reset_index()
        .drop("level_1", axis=1)
    )
