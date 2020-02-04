import numpy as np
import pandas as pd
import warnings
from ..alignment import align_to, negative_align, align_around


def align_df(
    df: pd.core.frame.DataFrame,
    data_colname: str,
    events: np.ndarray,
    max_latency: float = None,
    t_before: float = None,
    drop: bool = False,
):
    """
    Aligns a column in a df to events.
    params:
        df: the dataframe containing the data to be aligned
        data_colname: the column name in df to be aligned
        events: a series or numpy array of event timestamps to align to
        max_latency: if specified, any latencies above this will be returned as np.nan
        t_before: the desired negative window before the onset of the event to align to
        drop: whether to drop np.nan values
    returns:
        a pd.Series of values in df[t_colname] aligned to events 
    """
    return align_around(
        df[data_colname].values,
        to_align_to=events,
        t_before=t_before,
        max_latency=max_latency,
        drop=drop,
    )


def align_df_by(
    df_data: pd.core.frame.DataFrame,
    df_data_data_colname: str,
    df_data_group_colname: str,
    df_events: pd.core.frame.DataFrame,
    df_events_event_colname: str,
    df_events_group_colname: str,
    max_latency: float = None,
    t_before: float = None,
):
    """
    Aligns a data in a data df to events in a event df.

    params:
        df_data: the dataframe containing the data to be aligned
        df_data_data_colname: label of the column name in df_data 
                           containing data to be aligned 
        df_data_group_colname: label of the column in df_data containing 
                               group membership identifiers 
        df_events: dataframe containing events to align to
        df_events_event_colname: label of column in df_events containing events 
        df_events_group_colname: label of the column in df_events containing 
                                 group membership identifiers 
        max_latency: if specified, any latencies above this will be returned as np.nan
        t_before: the desired negative window before the onset of the event to align to
    returns:
        a copy of df_data with an additional column: aligned
    """
    if not (
        df_data[df_data_group_colname].dtype == df_events[df_events_group_colname].dtype
    ):
        raise TypeError(
            "Columns containing groups identifiers must be of the same type"
        )

    group_names_data = df_events[df_events_group_colname].unique()
    group_names_events = df_data[df_data_group_colname].unique()
    if group_names_data.tolist() == group_names_events.tolist():
        warnings.warn("Groups in df_events and df_data are not identical")
    frames = []
    for group_name in group_names_data:
        events = df_events[df_events[df_data_group_colname] == str(group_name)][
            df_events_event_colname
        ].values
        df = df_data[df_data[df_data_group_colname] == str(group_name)].copy()

        res = align_df(
            df,
            data_colname=df_data_data_colname,
            events=events,
            max_latency=max_latency,
            t_before=t_before,
            drop=False,
        )
        df["aligned"] = res
        frames.append(df)
    return pd.concat(frames, axis=0)
