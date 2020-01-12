import numpy as np
import pandas as pd


def bin_df(
    df: pd.core.frame.DataFrame,
    colname: str,
    bins: np.ndarray,
    return_idx=False,
    return_values=True,
    bin_idx_name: str = "bin_idx",
    bin_val_name: str = "bin_val",
):
    """
    Given a dataframe containing a continuous column, and bin values,
    appends the appropriate index and or value to the dataframe
    """
    # make this a numpy function in the original package
    df[bin_idx_name] = np.digitize(df[colname], bins) - 1
    if return_values:
        df[bin_val_name] = bins[df[bin_idx_name].tolist()]
    if not return_idx:
        df.drop(bin_idx_name, axis=1, inplace=True)
    return df
