import matplotlib.pyplot as plt
import numpy as np
from spiketimes.alignment import align_to, negative_align
from matplotlib.ticker import MaxNLocator
from ..alignment import align_around


def psth(
    spiketimes: np.ndarray,
    events: np.ndarray,
    binwidth: float = 0.01,
    t_before: float = 0.2,
    max_latency: float = 2,
    ax=None,
    hist_kwargs: dict = None,
):
    """Contruct a peristimulus time histogram of spike_times with respect to events
    t_before defines the time before time 0 (when the event occured) to include 
    in the histogram"""
    # TODO implement binwidth functionality

    if ax is None:
        _, ax = plt.subplots()

    latencies = align_around(spiketimes, events, t_before, max_latency)
    bins = np.arange(np.min(latencies), np.max(latencies), binwidth)

    if hist_kwargs is None:
        hist_kwargs = {}
    ax.hist(latencies, bins=bins, **hist_kwargs)
    ax = add_event_vlines(ax, 0)
    return ax


def add_event_vlines(
    ax,
    events,
    linestyle: str = "--",
    color: str = "grey",
    t_min: float = None,
    t_max: float = None,
    vline_kwargs: dict = None,
):
    """Add vertical lines at the point(s) specified in events
    t_min and t_max define minimum and maximum timepoints for events i.e. no
    events outside these limits will be plotted"""
    if vline_kwargs is None:
        vline_kwargs = {}
    try:
        _ = (x for x in events)
    except TypeError:
        events = [events]
    if t_min:
        events = np.array(list(filter(lambda x: x > t_min, events)))
    if t_max:
        events = np.array(list(filter(lambda x: x < t_max, events)))
    for event in events:
        ax.axvline(event, color=color, linestyle=linestyle, **vline_kwargs)
    return ax
