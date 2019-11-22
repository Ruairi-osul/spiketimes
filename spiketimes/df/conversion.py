import numpy as np
import pandas as pd


def list_to_df(spiketrains: list, indexes=None):
    """
    convert a list of spiketrains into a tidy dataframe of spiketimes
    params:
        spiketrains: list of spiketrains
        indexes: optional list of 
    """
    if indexes is None:
        indexes = np.arange(len(spiketrains))
    else:
        assert len(spiketrains) == len(
            indexes
        ), "index and spiketrains must be the same size"

    df_list = [
        pd.DataFrame({"spiketrain": index, "timepoint_s": spiketrain})
        for index, spiketrain in zip(indexes, spiketrains)
    ]

    return pd.concat(df_list)
