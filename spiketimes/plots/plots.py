import matplotlib.pyplot as plt
import numpy as np
from spiketimes.alignment import align_to, negative_align
from matplotlib.ticker import MaxNLocator


def psth(spike_times, events, t_before=0.2, max_latency=2, ax=None, **kwargs):
    """Contruct a peristimulus time histogram of spike_times with respect to events
    t_before defines the time before time 0 (when the event occured) to include 
    in the histogram"""
    # TODO implement binwidth functionality
    # TODO implement maximum latencies
    if ax is None:
        _, ax = plt.subplots()
    postive_latencies = align_to(spike_times, events, no_beyond=True)
    negative_latencies = negative_align(spike_times, events, no_before=True)
    latencies = np.concatenate(
        (
            negative_latencies[np.logical_not(np.isnan(negative_latencies))],
            postive_latencies[np.logical_not(np.isnan(postive_latencies))],
        )
    )
    latencies = latencies[(latencies >= -t_before) & (latencies <= max_latency)]

    ax.hist(latencies, **kwargs)
    return ax


def add_event_vlines(
    ax, events, linestyle="--", color="grey", t_min=None, t_max=None, **kwargs
):
    """Add vertical lines at the point(s) specified in events
    t_min and t_max define minimum and maximum timepoints for events i.e. no
    events outside these limits will be plotted"""
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
