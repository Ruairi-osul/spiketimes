import numpy as np
from ..alignment import align_to, negative_align


def _compare_pos_neg(row, t_before, total_cycle):
    """
    Private function which for each row, compares the values in columns
    pos and neg. Depending on these values, returns the aligned value
    """
    ret = np.nan
    ret = row["neg"] if row["neg"] >= t_before else ret
    ret = (
        row["pos"]
        if (np.isnan(ret)) and (row["pos"] <= (total_cycle + t_before))
        else ret
    )
    return ret


def align_df(df, t_colname, events, total_cycle, t_before):
    """
    Aligns a column in a df to events.
    params:
        df: the dataframe containing the data to be aligned
        t_colname: the column name in df to be aligned
        events: a series or numpy array of event timestamps to align to
        total_cycle: the total time period of the event being aligned to
        t_before: the desired negative window before the onset of the event to align to
    """
    return df.assign(
        pos=align_to(df[t_colname], events), neg=negative_align(df[t_colname], events),
    ).apply(_compare_pos_neg, t_before=t_before, total_cycle=total_cycle, axis=1,)
