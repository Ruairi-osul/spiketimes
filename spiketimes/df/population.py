import pandas as pd
import numpy as np
from scipy import stats
import spiketimes.correlate


def population_coupling_df(
    df: pd.core.frame.DataFrame,
    spiketrain_col: str = "spiketrain",
    spiketimes_col: str = "spiketimes",
    binsize: float = 0.01,
    num_lags: int = 100,
    t_start: float = None,
    t_stop: float = None,
    return_all: bool = False,
):
    """
    Calculate the population-coupling index between each spiketrain and all others in a DataFrame.

    The metric is calculated by computing and standardising cross correlation
    between an individual spiketrain and the "population spiketrain", consisting of all other neurons.
    Large Z score cross correlation at lag=0 is indicative of high population coupling.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        spiketrain_col: The column containing spiketimes
        spiketimes_col: The column containing spiketrain identifiers
        binsize: The size of the time bin in seconds
        num_lags: The number of lags forward and backwards around lag 0 to return
        t_start: Minimum timepoint
        t_stop: Maximum timepoint
        return_all: If true, all time bins and cross correlation values are returned
    Returns:
        A pandas DataFrame containing one row per spiketrain with columns {spiketrain_col, 'population_coupling'}
    """
    ROUNDING_PRECISION = 5
    FRAC_TO_COMPARE = 4

    bin_idx_to_start = ((num_lags * 2) + 1) // FRAC_TO_COMPARE
    out: list = []
    spiketrains = df[spiketrain_col].unique()
    for spiketrain in spiketrains:
        spiketrain_oi = df[df[spiketrain_col] == spiketrain][spiketimes_col].values
        population_spiketrain = np.sort(
            df[df[spiketrain_col] != spiketrain][spiketimes_col].values
        )
        t, cc = spiketimes.correlate.cross_corr(
            spiketrain_1=spiketrain_oi,
            spiketrain_2=population_spiketrain,
            binsize=binsize,
            num_lags=num_lags,
            as_df=False,
            t_start=t_start,
            t_stop=t_stop,
            delete_0_lag=False,
        )
        z = stats.zmap(cc, cc[:bin_idx_to_start])
        t = np.round(t, ROUNDING_PRECISION)
        if return_all:
            out.append(
                pd.DataFrame({"time_sec": t, "zscore": z, spiketrain_col: spiketrain})
            )
        else:
            out.append(z[t == 0])
    if return_all:
        df = pd.concat(out, axis=0)
    else:
        df = pd.DataFrame({spiketrain_col: spiketrains, "population_coupling": out})
    return df
