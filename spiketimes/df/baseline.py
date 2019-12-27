import pandas as pd


def create_baseline_df(
    df, timepoint_cutoff_max, timepoint_cutoff_min=None, timepoint_colname="timepoint"
):
    """Given a tidy spike dataframe, will return a df of the baseline period"""
    if timepoint_cutoff_min:
        return df.loc[
            (df["timepoint"] >= timepoint_cutoff_min)
            & (df["timepoint"] < timepoint_cutoff_max)
        ].pipe(lambda x: x)
    return df.loc[df["timepoint"] < timepoint_cutoff_max]


def zscore_normalise_by_neuron(
    df,
    timepoint_cutoff_max,
    timepoint_cutoff_min=None,
    col_to_act_on="firing_rate",
    timepoint_colname="timepoint",
):
    # create baseline df and get sd and mean
    # join with original and mutate for final
    pass

