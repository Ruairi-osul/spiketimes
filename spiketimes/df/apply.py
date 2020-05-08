import pandas as pd


def apply_by(
    df: pd.core.frame.DataFrame,
    func,
    func_kwargs: dict = None,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    returned_colname: str = "apply_result",
):
    """
    Apply an arbitrary function to each spiketrain in a DataFrame.

    The passed function should have a single return value for each spiketrain.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        func: The function to apply to the data
        func_kwargs: dictionary of key-word arguments to be passed to the function
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column containing spiketrain identifiers
        return_colname: The label of the column in the returned DataFrame containing the function result
    Returns:
        A pandas DataFrame with columns {spiketrian_col and returned_colname}
    """
    if not func_kwargs:
        func_kwargs = {}
    res = (
        df.groupby(spiketrain_col)
        .apply(lambda x: func(x[spiketimes_col].values, **func_kwargs))
        .reset_index()
        .rename(columns={0: returned_colname})
    )
    if "level_1" in res.columns:
        res = res.rename(columns={"level_1": f"{returned_colname}_idx"})
    return res


def apply_by_rolling(
    df: pd.core.frame.DataFrame,
    func,
    num_periods: int = 10,
    func_kwargs: dict = None,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    returned_colname: str = "rolling_result",
    copy: bool = True,
):
    """
    Apply a function in a roling window along each neuron in a dataframe

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        func: funtion to apply along the datafrmae
        num_period: The number of rows in the rolling window
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column containing spiketrain identifiers
        returned_colname: The label of the column in the returned DataFrame containing the function result
        copy: Whether make a copy of the passed to DataFrame before applying the function
    Returns:
        A copy of the passed DataFrame with returned_colname appended
    """
    original_index_name = df.index.name
    if not func_kwargs:
        func_kwargs = {}
    if copy:
        df = df.copy()
    tmp_res = (
        df.groupby(spiketrain_col)[spiketimes_col]
        .rolling(num_periods)
        .apply(lambda x: func(x.values, **func_kwargs), raw=True)
        .reset_index()
        .rename(columns={spiketimes_col: returned_colname})
        .set_index("level_1")
    )
    tmp_res.index.name = "index"
    tmp_res = pd.merge(df.reset_index(), tmp_res.reset_index()).set_index("index")
    tmp_res.index.name = original_index_name
    return tmp_res
