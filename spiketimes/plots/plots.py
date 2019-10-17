import matplotlib.pyplot as plt
import numpy as np
from spiketimes import align_to
from matplotlib.ticker import MaxNLocator


def psth(spike_times, events, t_before, ax=None, **kwargs):
    # TODO implement binwidth functionality
    if ax is None:
        _, ax = plt.subplots()
    latencies = align_to(spike_times, events) - t_before
    ax.hist(latencies, **kwargs)
    return ax


def add_event_vlines(
    ax, events, linestyle="--", color="grey", t_min=None, t_max=None, **kwargs
):
    try:
        _ = (x for x in events)
    except TypeError:
        events = [events]
    if t_min:
        events = np.array(list(filter(lambda x: x > t_min, events)))
    if t_max:
        events = np.array(list(filter(lambda x: x < t_max, events)))
    for event in events:
        ax.axvline(event, color=color, linestyle=linestyle, **kwargs)
    return ax
