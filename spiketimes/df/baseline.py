import pandas as pd


def create_baseline_df(
    df, timepoint_cutoff_max, timepoint_cutoff_min=None, timepoint_colname="timepoint"
):
    """Given a tidy spike dataframe, will return a df of the baseline period"""
    if timepoint_cutoff_max:
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
    pass


def apply_by_neuron(df):
    pass


def apply_by_neuron_rolling(
    df,
    num_periods,
    func,
    func_kwargs,
    col_to_act_on="firing_rate",
    neuron_col="neuron_id",
    returned_colname="Rolling_Result",
):
    tmp_res = (
        df.groupby(neuron_col)[col_to_act_on]
        .rolling(num_periods)
        .apply(lambda x: func(x, **func_kwargs), raw=False)
        .reset_index()
        .rename(columns={col_to_act_on: returned_colname})
        .set_index("level_1")
    )

    tmp_res.index.name = "index"

    return pd.merge(df.reset_index(), tmp_res.reset_index()).set_index("index")
