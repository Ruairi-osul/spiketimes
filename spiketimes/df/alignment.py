import numpy as np
import pandas as pd
import warnings
import spiketimes.alignment


def align_around(
    df: pd.core.frame.DataFrame,
    data_colname: str,
    events: np.ndarray,
    max_latency: float = None,
    t_before: float = None,
    drop: bool = False,
):
    """
    Aligns data to events.

    Two dataframes should be passed: one containing the data to be aligned and the other
    containing the event to align to. All data is aligned to the same events.
    Use spiketimes.df.alignment.align_around_by to align by group.

    Args:
        df: The dataframe containing the data to be aligned
        data_colname: The column name in df to be aligned
        events: A series or numpy array of event timestamps to align to
        max_latency: If specified, any latencies above this will be returned as np.nan
        t_before: The desired negative window before the onset of the event to align to
        drop: Whether to drop np.nan values
    Returns:
        A copy of df with an additional column `aligned` containing values in df[t_colname] aligned to events
    """
    return df.assign(
        aligned=spiketimes.alignment.align_around(
            df[data_colname].values,
            to_align_to=events,
            t_before=t_before,
            max_latency=max_latency,
            drop=drop,
        )
    )


def align_around_by(
    df_data: pd.core.frame.DataFrame,
    df_events: pd.core.frame.DataFrame,
    df_data_data_colname: str = "spiketimes",
    df_events_group_colname: str = "session",
    df_data_group_colname: str = "session",
    df_events_event_colname: str = "spiketimes",
    max_latency: float = None,
    t_before: float = None,
):
    """
    Align data to events. Align different datapoints to different events.

    Aligns data in a data in a pandas dataframe (df_data) to events in an event df (df_events).
    Data is aligned to events sharing the same group. Useful when aligned data from different
    sessions to events from different sessions.

    Args:
        df_data: A pandas DataFrame containing data to be aligned
        df_data_data_colname: The label of the column in df_data
                           containing the data to be aligned
        df_data_group_colname: The label of the column in df_data containing
                               group membership identifiers.
        df_events: the df containing events to the data align to
        df_events_event_colname: The label of column in df_events containing events
        df_events_group_colname: The label of the column in df_events containing
                                 group membership identifiers (e.g. session id).
        max_latency: If specified, any latencies above this will be returned as np.nan
        t_before: The desired negative window before the onset of the event to align to
    Returns:
        A copy of df_data with an additional column: 'aligned' containing data aligned to events.
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
        res = align_around(
            df,
            data_colname=df_data_data_colname,
            events=events,
            max_latency=max_latency,
            t_before=t_before,
            drop=False,
        )["aligned"].values
        df["aligned"] = res
        frames.append(df)
    return pd.concat(frames, axis=0)
