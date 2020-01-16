import pandas as pd
from ..surrogates import shuffled_isi_spiketrains
from .conversion import list_to_df


def shuffled_isi_spiketrains_df(
    df: pd.core.frame.DataFrame, spiketrain_col: str = "spiketimes", n: int = 1
):
    return list_to_df(shuffled_isi_spiketrains(df[spiketrain_col]))


def shuffled_isi_spiketrains_df_by(
    df: pd.core.frame.DataFrame,
    spiketrain_col: str = "spiketimes",
    by_col: str = "neuron_id",
    n: int = 2000,
):
    return (
        df.groupby(by_col)
        .apply(
            lambda x: shuffled_isi_spiketrains_df(x, spiketrain_col=spiketrain_col, n=n)
        )
        .reset_index()
        .drop("level_1", axis=1)
        .rename(columns={"spiketrain": "spiketrain_replicate"})
    )
