import pandas as pd
import numpy as np
import spiketimes.df.binning


def zscore_standardise_by(
    df: pd.core.frame.DataFrame,
    baseline_start_stop: np.ndarray,
    spiketrain_col: str = "spiketrain",
    time_col: str = "time",
    data_col: str = "spike_count",
    returned_colname: str = "zscore",
):
    """
    For each spiketrain, convert a data column to zscores using only data from the baseline period.

    Args:
        df: A pandas DataFrame containing multiple data points per spiketrain
        baseline_start_stop: A numpy array containing the starting and ending time of the baseline
                             period.
        spiketrain_col: The column containing spiketrain identifiers
        time_col: The column containing time points
        data_col: The column containing data to be zscore standardised
        returned_colname:
    Returns:
        A copy of the passed DataFrame with an additional column containing zscores
    """
    dfb = _create_baseline_df(
        df, baseline_start_stop, data_col="spike_count", time_col="time"
    )
    dfb = (
        dfb.groupby(spiketrain_col)
        .apply(
            lambda x: pd.Series({"mean": x[data_col].mean(), "std": x[data_col].std()})
        )
        .reset_index()
    )
    df = pd.merge(df, dfb, on="spiketrain")
    return df.assign(
        **{returned_colname: df[data_col].subtract(df["mean"]).divide(df["std"])}
    ).drop(["mean", "std"], axis=1)


def _create_baseline_df(df, baseline_start_stop: np.ndarray, data_col, time_col="time"):
    """
    Subset a pandas DataFrame to contain only data from the baseline period.
    """
    df = (
        spiketimes.df.binning.which_bin(
            df=df, spiketimes_col=time_col, bin_edges=baseline_start_stop
        )
        .drop("bin_values", axis=1)
        .rename(columns={"bin_idx": "is_baseline"})
    )
    df["is_baseline"] = df["is_baseline"].map({0: 1, 1: 0})
    return df[df["is_baseline"] == 1].drop("is_baseline", axis=1)
