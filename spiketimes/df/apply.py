import pandas as pd


def apply_by_neuron(
    df: pd.core.frame.DataFrame,
    func,
    col_to_act_on: str,
    func_kwargs: dict = {},
    neuron_col: str = "neuron_id",
    returned_colname: str = "apply_result",
):
    """
    Apply a function to each neuron individually and aggregate the results
    into an dataframe
    
    params:
        df: pandas.DataFrame to act on
        func: function to apply
        col_to_act_on: label of column containg the data to be passed to
                       the function
        func_kwargs: dictionary of key word arguments to be passed to the
                     function
        neuron_col: label of the column in df which containing neuron identifiers
        return_colname: label of the column to return contained the function result
    returns:
        pandas dataframe with columns: neuron_col and returned_colname
    """
    return (
        df.groupby(neuron_col)
        .apply(lambda x: func(x[col_to_act_on], **func_kwargs))
        .reset_index()
        .rename(columns={0: returned_colname})
    )


def apply_by_neuron_rolling(
    df: pd.core.frame.DataFrame,
    func,
    num_periods: int = 10,
    func_kwargs: dict = {},
    col_to_act_on: str = "firing_rate",
    neuron_col: str = "neuron_id",
    returned_colname: str = "rolling_result",
):
    """
    Apply a function in a roling window along each neuron in a dataframe

    params:
        df: pandas dataframe containing the data
        func: funtion to apply along the datafrmae
        num_period: number of rows in the rolling window
        neuron_col: label of column containing neuron_identifiers
        returned_colname: label of returned column containing the output of
                          function
    returns:
        the original dataframe with returned_colname appended
    """
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

