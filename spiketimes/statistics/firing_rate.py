from ..alignment import binned_spiketrain
from elephant.statistics import sskernel
from scipy.ndimage import gaussian_filter1d
import pandas as pd


def ifr(spiketimes, fs, t_start=None, t_stop=None, sigma=None, as_df=True):
    if t_start is None:
        t_start = spiketimes[0]
    if t_stop is None:
        t_stop = spiketimes[-1]
    df = binned_spiketrain(spiketimes, fs, t_stop=t_stop, t_start=t_start, as_df=True)
    df["spike_count"] = df["spike_count"].divide(1 / fs)
    if sigma is None:
        sigma = sskernel(spiketimes, tin=None, bootstrap=False)["optw"]
    smoothed = gaussian_filter1d(df["spike_count"], sigma)
    if not as_df:
        return df["time"].values, smoothed
    else:
        return pd.DataFrame({"time": df["time"], "ifr": smoothed})

