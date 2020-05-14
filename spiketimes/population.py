from .correlate import cross_corr
import numpy as np
from scipy.stats import zmap


def population_coupling(
    spiketrain: np.ndarray,
    spiketrain_list: list,
    bin_window: float = 0.01,
    num_lags: int = 100,
    as_df: bool = False,
    t_start: float = None,
    t_stop: float = None,
    frac_zscore: float = 0.25,
    return_all: bool = False,
):
    """
    Calculate the population-coupling index between a spiketrain and the population.

    The metric is calculated by computing and standardising cross correlation
    between an individual spiketrain and the "population spiketrain", consisting of all other neurons.
    Large Z score cross correlation at lag=0 is indicative of high population coupling.

    Args:
        spiketrain: A numpy array of spiketimes
        spiketrain_list: A list of numpy-array spiketrains
        binsize: The size of the time bin in seconds
        num_lags: The number of lags forward and backwards around lag 0 to return
        as_df: Whether to return results as pandas DataFrame
        t_start: Minimum timepoint
        t_stop: Maximum timepoint
        return_all: If true, all time bins and cross correlation values are returned
    Returns:
        The zscore at lag=0 between the spiketrain and the population
    """
    T_ROUNDING_PRECISION = 5

    t_cutoff = ((num_lags * 2) + 1) // (1 / frac_zscore)
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
    time_bins = np.round(time_bins, T_ROUNDING_PRECISION)
    values = zmap(values, values[:t_cutoff])
    if not return_all:
        return values[time_bins == 0]
    else:
        return time_bins, values
