import pandas as pd
import numpy as np


def create_baseline_df(
    df, timepoint_cutoff_max, timepoint_cutoff_min=None, timepoint_colname="timepoint"
):
    """Given a tidy spike dataframe, will return a df of the baseline period"""
    if timepoint_cutoff_min:
        return df.loc[
            (df[timepoint_colname] >= timepoint_cutoff_min)
            & (df[timepoint_colname] < timepoint_cutoff_max)
        ].pipe(lambda x: x)
    return df.loc[df[timepoint_colname] < timepoint_cutoff_max]


def _mean_std(df, col_to_act_on):
    return pd.Series(
        {"mean": np.mean(df[col_to_act_on]), "std": np.std(df[col_to_act_on])}
    )


def zscore_normalise_by_neuron(
    df,
    timepoint_cutoff_max,
    timepoint_cutoff_min=None,
    col_to_act_on="firing_rate",
    timepoint_colname="timepoint",
    neuron_id_colname="neuron_id",
    new_colname="zscore",
):
    # TODO: convert to zmap
    # create baseline df
    dfn = create_baseline_df(
        df,
        timepoint_cutoff_max=timepoint_cutoff_max,
        timepoint_cutoff_min=timepoint_cutoff_min,
        timepoint_colname=timepoint_colname,
    )
    # get sd and mean
    dfn = (
        dfn.groupby(neuron_id_colname)
        .apply(_mean_std, col_to_act_on=col_to_act_on)
        .reset_index()
    )

    # join with original
    df = pd.merge(dfn, df, on=neuron_id_colname)

    # mutate for final
    df = df.assign(temp=(df[col_to_act_on].subtract(df["mean"])).divide(df["std"]))
    df.drop(["mean", "std"], axis=1, inplace=True)
    df.rename(columns={"temp": new_colname}, inplace=True)
    return df

