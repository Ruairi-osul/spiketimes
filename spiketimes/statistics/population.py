from .autro_cross_corr import cross_corr
import numpy as np
import pandas as pd
from scipy.stats import zscore


def population_coupling(
    spiketrain,
    spiketrain_list: list,
    bin_window: float = 0.01,
    num_lags: int = 100,
    as_df: bool = False,
    t_start: float = None,
    t_stop: float = None,
):
    if t_start is None:
        t_start = spiketrain[0]
    if t_stop is None:
        t_stop = spiketrain[-1]
    population_spiketrain = np.sort(np.concatenate(spiketrain_list))
    time_bins, values = cross_corr(
        spiketrain,
        population_spiketrain,
        bin_window=bin_window,
        num_lags=num_lags,
        as_df=False,
        t_start=t_start,
        t_stop=t_stop,
        delete_0_lag=False,
    )
    values = zscore(values)
    if not as_df:
        return time_bins, values
    else:
        return pd.DataFrame(
            {"time_sec": time_bins, "population_coupling_zscore": values}
        )

