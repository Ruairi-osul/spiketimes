import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def raster_single_neuron(spike_times, ax=None, y_data_ind=1, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    y_data = np.ones(shape=(1, len(spike_times))).flatten() + y_data_ind
    ax.scatter(spike_times, y_data, marker="|", **kwargs)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return ax


def raster_multiple_neurons(spike_times, ax=None, t_min=None, t_max=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    if t_min:
        spike_times = [
            np.array(list(filter(lambda x: x > t_min, spikes)))
            for spikes in spike_times
        ]
    if t_max:
        spike_times = [
            np.array(list(filter(lambda x: x < t_max, spikes)))
            for spikes in spike_times
        ]
    if isinstance(spike_times, list):
        for i, spikes in enumerate(spike_times):
            ax = raster_single_neuron(spikes, ax=ax, y_data_ind=i, **kwargs)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return ax
