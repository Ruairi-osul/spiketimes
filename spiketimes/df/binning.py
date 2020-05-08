import numpy as np
import pandas as pd
import spiketimes.binning
import warnings


def binned_spiketrain(
    df: pd.core.frame.DataFrame,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    fs: str = 1,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Get event counts by entity at regular a constant sampling rate.

    Args:
        df: Pandas dataframe containing the data
        fs: Desired sampling frequency in seconds
        spiketimes_col: The label of the column in df containing spiketimes
        spiketrain_col: The label of the column in df containing spiketrain identifiers.
        t_start: The time after which the first bin will start. Default is 0.
        t_stop: The maximum time for the time bins.
    Returns:
        A pandas DataFrame containing the binned data. The `time` column contains the left edge of the time bin.
        `spike_count` contains the number of spikes occuring in that bin.
    """
    if t_stop is None:
        t_stop = df[spiketimes_col].values[-1]

    return (
        df.groupby(spiketrain_col)
        .apply(
            lambda x: spiketimes.binning.binned_spiketrain(
                x[spiketimes_col], fs=fs, t_start=t_start, t_stop=t_stop, as_df=True
            )
        )
        .reset_index()
        .drop("level_1", axis=1)
    )


def binned_spiketrain_bins_provided(
    df: pd.core.frame.DataFrame,
    bins: np.ndarray,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
):
    """
    Get event count per item in user-specified bins.

    Designed to bin spiketrains but works on any set of events.

    Args:
        df: A pandas DataFrame containing the data
        bins: A numpy array of time bins
        spiketimes_col: The label of the column in df containing spiketimes
        spiketrain_col: The label of the column in df containing spiketrain identifiers.
    Returns:
        A pandas DataFrame with columns indicating the unit (`by_col`), time bin and event counts.
    """
    return (
        df.groupby(spiketrain_col)
        .apply(
            lambda x: pd.DataFrame(
                {
                    "bin": bins[:-1],
                    "counts": spiketimes.binning.binned_spiketrain_bins_provided(
                        x[spiketimes_col], bins=bins
                    ),
                }
            )
        )
        .reset_index()
        .drop("level_1", axis=1)
    )


def which_bin(
    df: pd.core.frame.DataFrame,
    bin_edges: np.ndarray,
    allow_before: bool = False,
    max_latency: float = None,
    before: float = None,
    spiketimes_col: str = "spiketimes",
):
    """
    Returns the closest bin for each data element. Useful for asigning spikes to trials.

    Args:
        df: A pandas DataFrame containing the data to be binned
        bin_edges: A numpy array of edges to bin into.
        before: If specified, the spiketrain is aligned to the bins
                shifts bins backwards by this quantity.
        allow_before: If False, spikes occuring before the first time bin return np.nan
        max_latency: If specified, np.nan is returned for any spikes occuring this quantity
                     after the maximum bin_edge
        spiketimes_col: The label of the column in df containing spiketimes
    Returns:
        A copy of the passed DataFrame with an additional two columns: 'bin_values' and 'bin_idx' containing the
        value and index in corresponding event array of the appropriate event.
    """
    return df.assign(
        bin_idx=lambda x: spiketimes.binning.which_bin(
            x[spiketimes_col].values,
            bin_edges=bin_edges,
            max_latency=max_latency,
            before=before,
            allow_before=allow_before,
        )[0],
        bin_values=lambda x: spiketimes.binning.which_bin(
            x[spiketimes_col].values,
            bin_edges=bin_edges,
            max_latency=max_latency,
            before=before,
            allow_before=allow_before,
        )[1],
    )


def which_bin_by(
    df_data: pd.core.frame.DataFrame,
    df_data_data_colname: str,
    df_data_group_colname: str,
    df_events: pd.core.frame.DataFrame,
    df_events_event_colname: str,
    df_events_group_colname: str,
    max_latency: float = None,
    before: float = None,
    allow_before: bool = False,
):
    """
    Get corresponding bin per data point. Searches bins by group.

    Args:
        df_data: the df containing the data to be binned
        df_data_data_colname: label of the column in df_data
                           containing the data to be binned
        df_data_group_colname: label of the column in df_data containing
                               group membership identifiers. This could be session id,
                               mouse id etc.
        df_data_spiketrain_colname: label of the column in df_data containing spiketrain id (could also be event_type)
        df_events: the df containing events to the data align to
        df_events_event_colname: label of the column in df_events containing events
        df_events_group_colname: label of the column in df_events containing
                                 group membership identifiers (e.g. session id).
        max_latency: if specified, any latencies above this will be returned as np.nan
        before: the desired negative window before the onset of the event to align to
        allow_before: if true allows for negative idx
    Returns:
        A copy of df_data with an additional two columns: 'bin_values' and 'bin_idx' containing the value and
        index in corresponding event array of the appropriate event.
    """
    if not (
        df_data[df_data_group_colname].dtype == df_events[df_events_group_colname].dtype
    ):
        raise TypeError(
            "Columns containing groups identifiers must be of the same type"
        )

    group_names_data = df_events[df_events_group_colname].unique()
    group_names_events = df_data[df_data_group_colname].unique()
    if not set(group_names_data.tolist()) == set(group_names_events.tolist()):
        warnings.warn("Groups in df_events and df_data are not identical")
    frames = []
    for group_name in group_names_data:
        events = df_events[df_events[df_data_group_colname] == group_name][
            df_events_event_colname
        ].values
        df = df_data[df_data[df_data_group_colname] == group_name].copy()
        res = which_bin(
            df,
            spiketimes_col=df_data_data_colname,
            bin_edges=events,
            max_latency=max_latency,
            before=before,
            allow_before=allow_before,
        )
        df["bin_idx"] = res["bin_idx"]
        df["bin_values"] = res["bin_values"]
        frames.append(df)
    return pd.concat(frames, axis=0)


def spike_count_around_event(
    df: pd.core.frame.DataFrame,
    events: np.ndarray,
    binsize: float,
    spiketimes_col: str = "spiketimes",
    by_col: str = "spiketrain",
):
    """
    Get spike counts for each neuron following events.

    Args:
        df: A pandas DataFrame containing the spike data.
        events: A numpy array of event timings.
        binsize: The timeperiod after each event during which spikes are counted.
        spiketimes_col: The label of the column in df containing spiketimes.
        by_col: The label of the column in df containing spiketrain identifiers.
    Returns:
        A pandas DataFrame with columns identifing the spiketrain, event and spikecounts.
    """
    return (
        df.groupby(by_col)
        .apply(
            lambda x: pd.DataFrame(
                {
                    "event": events,
                    "counts": spiketimes.binning.spike_count_around_event(
                        x[spiketimes_col].values, events=events, binsize=binsize
                    ),
                }
            )
        )
        .reset_index()
        .drop("level_1", axis=1)
    )


def spike_count_around_event_by(
    df_data: pd.core.frame.DataFrame,
    binsize: float,
    df_data_data_colname: str,
    df_data_group_colname: str,
    df_data_spiketrain_colname: str,
    df_events: pd.core.frame.DataFrame,
    df_events_event_colname: str,
    df_events_group_colname: str,
):
    """
    Get spike counts around events where you different sets of spiketrains and events.

    Args:
        df_data: A pandas DataFrame containing the spike times
        binsize: The duration of the period after each event during which spikes are counted
        df_data_data_colname: The label of the column in df_data containing the spiketime data
        df_data_group_colname: The label of the column in df_data containing the group data (e.g. session_id)
        df_data_spiketrain_colname: The label of the column in df_data containg spiketrain ids
        df_events: A pandas DataFrame containing event timings
        df_events_event_colname: The label of the column in df_events containing event timings
        df_events_group_colname: The label of the column in df_events containing group identifiers (e.g. session_id).
    Returns:
        A pandas DataFrame with one row per event per spiketrain with columns identifying the event,
        spike counts, group and spiketrain.
    """
    if not (
        df_data[df_data_group_colname].dtype == df_events[df_events_group_colname].dtype
    ):
        raise TypeError(
            "Columns containing groups identifiers must be of the same type"
        )

    group_names_data = df_events[df_events_group_colname].unique()
    group_names_events = df_data[df_data_group_colname].unique()
    if not set(group_names_data.tolist()) == set(group_names_events.tolist()):
        warnings.warn("Groups in df_events and df_data are not identical")
    frames = []
    for group_name in group_names_data:
        events = df_events[df_events[df_data_group_colname] == group_name][
            df_events_event_colname
        ].values
        df = df_data[df_data[df_data_group_colname] == group_name].copy()
        res = spike_count_around_event(
            df=df,
            binsize=binsize,
            events=events,
            spiketimes_col=df_data_data_colname,
            by_col=df_data_spiketrain_colname,
        ).assign(**{df_data_group_colname: group_name})
        frames.append(res)
    return pd.concat(frames, axis=0)
