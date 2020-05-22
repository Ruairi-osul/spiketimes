import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import warnings
from .alignment import align_around, split_by_trial


def _raster(spiketrain: np.ndarray, ax=None, y_data_ind: int = 1, **kwargs):
    """
    Construct a raster plot of a single spiketrain over one trial

    Args:
        spiketrain: A numpy array of spiketimes in seconds
        ax: A matplotlib axes object to plot on
        y_data_ind: The y tick for spiketrain
        kwargs: Kwargs to pass to matplotlib.pyplot.plot
    Returns:
        A matplotlib axes object
    """
    try:
        (_ for _ in spiketrain[0])
        raise TypeError(
            f"Must Pass in a single numpy array. Nested iterable found.\n"
            f"Spike times: {spiketrain}"
        )
    except TypeError:
        pass
    if ax is None:
        _, ax = plt.subplots()
    y_data = np.zeros(shape=(1, len(spiketrain))).flatten() + y_data_ind
    ax.scatter(spiketrain, y_data, marker="|", **kwargs)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yticks([y_data_ind])
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Spiketrain")
    return ax


def raster(
    spiketrain_list: list,
    ax=None,
    skip_empty=True,
    t_start: float = None,
    t_stop: float = None,
    _starting_ytick=None,
    **kwargs,
):
    """
    Construct a raster plot of multiple spiketrains

    Args:
        spiketrain_list: A list of numpy arrays containing the timings of spiking events
        ax: A matplotlib axes object on which to plot
        skip_empty: Whether to skip spiketrains with no spikes in the plotting interval 
        t_start: Minimum timepoint
        t_stop: Maximum timepoint
        _starting_y_tick: The position on the y axis to start. 
        kwargs: Additional key-word arguments will be passed into matplotlib.pyplot.plot
    Returns:
        A matloblib axes object
    """
    if _starting_ytick is None:
        _starting_ytick = 0

    try:
        (_ for _ in spiketrain_list[0])  # iterable check
        if len(spiketrain_list) == 1:
            raise TypeError
    except TypeError:
        raise TypeError(
            f"spike_time must be an iterable containing at least array of spiketimes\n"
            f"Passed spiketimes: {spiketrain_list}"
        )

    if ax is None:
        _, ax = plt.subplots()
    if t_start:
        spiketrain_list = [
            np.array(list(filter(lambda x: x > t_start, spikes)))
            for spikes in spiketrain_list
        ]
    if t_stop:
        spiketrain_list = [
            np.array(list(filter(lambda x: x < t_stop, spikes)))
            for spikes in spiketrain_list
        ]
    if isinstance(spiketrain_list, list):
        i = 0
        for spikes in spiketrain_list:
            if len(spikes):
                ax = _raster(spikes, ax=ax, y_data_ind=i + _starting_ytick, **kwargs)
                i += 1
            else:
                warnings.warn(
                    "A spiketrain with no spikes in the plotting window was passed, skipping."
                )
                i += 1 if skip_empty else 0
    else:
        raise ValueError("Must pass in a list of spike times")

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return ax


def grouped_raster(
    st_lists: list,
    color_list: list = None,
    ax=None,
    skip_empty=True,
    t_start: float = None,
    t_stop: float = None,
    plot_kwargs=None,
    space_between_groups: int = 2,
):
    """
    Construct a raster plot of multiple groups of spiketrains.

    Args:
        st_lists: A list of lists of spiketrains. Each sublist contains one group of spiketrains.
        color_list: A list containing one color for each group to be plotted/
        ax: A matplotlib axes object
        skip_empty: Whether to skip spiketrains with no spikes in the plotting interval 
        t_start: Minimum timepoint
        t_stop: Maximum timepoint
        plot_kwargs: Additional key-word arguments will be passed into matplotlib.pyplot.plot
        space_between_groups: Number of spaces between groups in the y direction.
    Returns:
        A matloblib axes object
    """
    DEFAULT_COLORS = [
        "black",
        "red",
        "green",
        "blue",
        "pink",
        "purple",
        "orange",
        "yellow",
    ]
    if color_list is None:
        color_list = DEFAULT_COLORS
    if ax is None:
        _, ax = plt.subplots()
    if plot_kwargs is None:
        plot_kwargs = {}
    starting_ytick = 0
    for i, st_list in enumerate(st_lists):
        k1 = plot_kwargs.copy()
        k1["color"] = color_list[i] if i < len(color_list) else color_list[i - i]
        ax = raster(
            st_list,
            ax=ax,
            t_start=t_start,
            t_stop=t_stop,
            _starting_ytick=starting_ytick,
            **k1,
        )
        starting_ytick += len(st_list) + space_between_groups
    return ax


def aligned_raster(
    spiketrain: np.ndarray,
    trial_starts: np.ndarray,
    before: float = None,
    max_latency: float = None,
    ax=None,
    raster_kwargs=None,
):
    """
    Constructs a raster plot with each row containing spikes from a single trial.

    Args:
        spiketrain: a spiketrain containing spiketimes in seconds.
        trial_starts: an array of trial starts in seconds.
        before: if specified, include this amount of time (in seconds) before each trial
        max_latency: if specified, exclude spikes occuring this amount of time (in seconds)
                     after the final event.
        ax: matplotlib axes object to plot on.
    Returns:
        A matplotlib axes object
    """
    st_list = split_by_trial(
        spiketrain=spiketrain,
        trial_starts=trial_starts,
        max_latency=max_latency,
        before=before,
        kwargs=raster_kwargs,
    )
    ax = raster(st_list, ax=ax)
    ax.set_ylabel("Trial")
    return ax


def psth(
    spiketimes: np.ndarray,
    events: np.ndarray,
    binwidth: float = 0.01,
    t_before: float = 0.2,
    max_latency: float = 2,
    ax=None,
    hist_kwargs: dict = None,
    vline_kwargs: dict = None,
):
    """
    Contruct a peristimulus time histogram of spiketimes latencies to events.

    t_before defines the time before time 0 (when the event occured) to include
    in the histogram

    Args:
        spiketimes: A numpy array of spiketimes
        events: A numpy array of event times in the same units as spiketimes
        binwidth: The width of time bins
        t_before: The time before the aligned event to include in the psth
        max_latency: The maximum allowed latency. Useful for excluding spikes occuring after the final event.
        ax: An optional matloblib axes object to use
        hist_kwargs: A dict of kwargs to pass to matplotlib.pyplot.hist
    Returns:
        A matplotlib.pyplot.axes object
    """

    if ax is None:
        _, ax = plt.subplots()
    if hist_kwargs is None:
        hist_kwargs = {}
    if vline_kwargs is None:
        vline_kwargs = {}
    hist_kwargs["alpha"] = 0.5 if "alpha" not in hist_kwargs else hist_kwargs["alpha"]
    vline_kwargs["linewidth"] = (
        2.5 if "linewidth" not in vline_kwargs else vline_kwargs["linewidth"]
    )
    latencies = align_around(spiketimes, events, t_before, max_latency, drop=True)
    bins = np.arange(np.min(latencies), np.max(latencies), binwidth)

    if hist_kwargs is None:
        hist_kwargs = {}
    ax.hist(latencies, bins=bins, **hist_kwargs)
    ax = add_event_vlines(ax, 0, vline_kwargs=vline_kwargs)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Counts")
    return ax


def add_event_vlines(
    ax,
    events: np.ndarray,
    t_min: float = None,
    t_max: float = None,
    vline_kwargs: dict = None,
):
    """
    Add vertical lines to a matplotlib axes object at the point(s) specified in events.
    t_min and t_max define minimum and maximum timepoints for events i.e. no
    events outside these limits will be plotted.

    Args:
        ax: the axes to plot on top of
        events: an array of points on the x axis to plot
        t_min: if specified, no points less than this will be plotted
        t_max: if specified, no points greater than this will be plotted
    Returns:
        matplotlib axes
    """
    if vline_kwargs is None:
        vline_kwargs = {}
    vline_kwargs["color"] = (
        "black" if "color" not in vline_kwargs else vline_kwargs["color"]
    )
    vline_kwargs["linestyle"] = (
        "--" if "linestyle" not in vline_kwargs else vline_kwargs["linestyle"]
    )
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
        ax.axvline(event, **vline_kwargs)
    return ax
