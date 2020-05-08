import numpy as np
import pandas as pd


def list_to_df(
    spiketrains: list,
    indexes: list = None,
    returned_spiketimes_label: str = "spiketimes",
    returned_spiketrain_label: str = "spiketrain",
):
    """
    Convert a list of spiketrains into a tidy dataframe of spiketimes

    Args:
        spiketrains: A list of numpy-array spiketrains
        indexes: An optional list of labels for the of the spiketrains
        returned_spiketimes_label: The label of the column in the returned DataFrame containing spiketimes
        returned_spiketrains_label: The label of the column in the returned DataFrame containing spiketrain identifiers
    Returns:
        A pandas DataFrame containing one spike and id label per row
    """
    if indexes is None:
        indexes = np.arange(len(spiketrains))
    else:
        assert len(spiketrains) == len(
            indexes
        ), "index and spiketrains must be the same size"

    df_list = [
        pd.DataFrame(
            {returned_spiketrain_label: index, returned_spiketimes_label: spiketrain}
        )
        for index, spiketrain in zip(indexes, spiketrains)
    ]

    return pd.concat(df_list)


def list_of_dicts_to_df(
    list_of_dicts: list,
    returned_spiketimes_label: str = "spiketimes",
    returned_spiketrain_label: str = "spiketrain",
):
    """
    Convert a list of named spiketrains to a dataframe.

    Data must be in the format of [{"st_name": spiketimes_arr}, etc]

    Args:
        list_of_dicts: A list of dicts. The dict key is the spiketrain identifier. The dict
                       value is a numpy array of spiketimes.
        returned_spiketimes_label: The label of the column in the returned DataFrame containing spiketimes
        returned_spiketrains_label: The label of the column in the returned DataFrame containing spiketrain identifiers
    Returns:
        A pandas DataFrame containing spiketimes indexed by spiketrain.
    """
    indexes = []
    spiketrains = []
    for st in list_of_dicts:
        for i, (k, v) in enumerate(st.items()):
            if i == 1:
                raise ValueError(
                    'Must pass a list of dictionaries of the form [{"stname": st_arr}'
                )
            indexes.append(k)
            spiketrains.append(v)
    return list_to_df(
        spiketrains=spiketrains,
        indexes=indexes,
        returned_spiketrain_label=returned_spiketrain_label,
        returned_spiketimes_label=returned_spiketimes_label,
    )


def df_to_list_of_dicts(
    df: pd.core.frame.DataFrame,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
):
    """
    Convert a DataFrame of spiketrains to a list of dicts of spiketrains.

    The dicts are of the form {spiketrain_id: spiketimes_array}

    Args:
        df: A pandas DataFrame of spiketimes indexed by spiketrains
        spiketimes_col: The column containing spiketimes
        spiketrain_col: The column containing spiketrain identifiers
    Returns:
        A list of dictionarys of the form {spiketrain_id: spiketimes_array}
    """
    return [
        {d[spiketrain_col]: d[spiketimes_col]}
        for d in df.groupby(spiketrain_col)[spiketimes_col]
        .apply(np.array)
        .reset_index()
        .to_dict(orient="records")
    ]


def df_to_list(
    df: pd.core.frame.DataFrame,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
):
    """
    Convert a DataFrame of spiketimes to a list spiketrains.

    Args:
        df: A pandas DataFrame of spiketimes indexed by spiketrains
        spiketimes_col: The column containing spiketimes
        spiketrain_col: The column containing spiketrain identifiers
    Returns:
        spiketrain_IDs, spiketrain_list
    """
    grouped = df.groupby(spiketrain_col)[spiketimes_col].apply(np.array)
    return grouped.index.values, grouped.tolist()
