from .binning import bin_df
import pandas as pd
import numpy as np


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
    
    Args:
        df: pandas.DataFrame to act on
        func: function to apply
        col_to_act_on: label of column containg the data to be passed to
                       the function
        func_kwargs: dictionary of key word arguments to be passed to the
                     function
        neuron_col: label of the column in df which containing neuron identifiers
        return_colname: label of the column to return contained the function result
    Returns:
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

    Args:
        df: pandas dataframe containing the data
        func: funtion to apply along the datafrmae
        num_period: number of rows in the rolling window
        neuron_col: label of column containing neuron_identifiers
        returned_colname: label of returned column containing the output of
                          function
    Returns:
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


def apply_by_neuron_by_bin(
    df: pd.core.frame.DataFrame,
    bins: np.ndarray,
    func,
    col_to_act_on: str,
    time_column: str,
    neuron_col: str = "neuron_id",
    func_kwargs: dict = {},
    returned_colname: str = "value",
):
    """
    Apply a function to each neuron at specified time bins.

    Args:
        df: dataframe containing the data
        bins: numpy array of time bins
        func: func to call on the data
        func_kwargs: optional dictionary of key-word-arguments to pass to func
        col_to_act_on: label of column in df containing data to be passed to func
        time_column: label of column in df with time info to be used for binning
        neuron_col: label of column in df with neuron identifiers
        returned_colname: optional label of returned column to pass 
    returns:
         pandas dataframe containing the results
    """
    return (
	    bin_df(df=df, colname=time_column, bins=bins, bin_val_name="bin")
	    .groupby([neuron_col, "bin"])
	    .apply(lambda x: func(x[col_to_act_on].values, **func_kwargs))
	    .unstack()
	    .reset_index()
	    .melt(id_vars=neuron_col, value_name=returned_colname)
            .sort_values(neuron_col)
            .reset_index(drop=True)
    )
