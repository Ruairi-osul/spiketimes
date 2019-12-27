import numpy as np


def bin_df(
    df, colname, bins, return_idx=False, return_values=True, bin_val_name="bin_val"
):
    df["bin_idx"] = np.digitize(df[colname], bins) - 1
    if return_values:
        df[bin_val_name] = bins[df["bin_idx"].tolist()]
    if not return_idx:
        df.drop("bin_idx", axis=1, inplace=True)
    return df
