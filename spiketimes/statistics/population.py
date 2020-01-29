from .autro_cross_corr import cross_corr
import numpy as np
import pandas as pd
from scipy.stats import zscore


def population_coupling(
    spiketrain: np.ndarray,
    spiketrain_list: list,
    bin_window: float = 0.01,
    num_lags: int = 100,
    as_df: bool = False,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Given a spiketrain and a list of simultaneously recorded spiketrains, 
    calculates a metric of population spike coupling for that spiketrain. 
    The metric is  calculated by calculating and zscore standardising cross correlation 
    between an individual spike train and the merged spiketrain of all other neurons. 
    Large Z score cross correlation at lag=0 is indicative of high population coupling.

    params:
        spiketrain: a numpy array of spiketimes in seconds 
        spiketrain_list: a list of numpy arrays containing spike times of individual 
                         simultaneously-recorded neurons.
        bin_window: size of time bin to use when calculating crosscorrelation
        num_lags: number of lags to shift forwards and backwargs for crosscorrelation
        as_df: whether to return the result as a pandas DataFrame
        t_start: if specified, spikes before this limit will not be consisdered
        t_stop: if specified, spikes after this limit will not be considered
    """
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

