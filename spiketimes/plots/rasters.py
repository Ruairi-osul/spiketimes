import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def raster_single_neuron(
    spiketrain: np.ndarray, ax=None, y_data_ind: int = 1, **kwargs
):
    """
    Construct a raster plot of a single spiketrain over one trial

    params:
        spiketrain: numpy array of spiketimes in seconds
        ax: matplotlib axes object to plot on
        y_data_ind: y tick for spiketrain
        kwargs: kwargs to pass to matplotlib.pyplot.plot
    returns:
        matplotlib axes
    """
    # TODO implement t_min and tmax
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
    return ax


def raster_multiple_neurons(
    spiketrain: list, ax=None, t_min: float = None, t_max: float = None, **kwargs
):
    """
    Construct a raster plot of multiple spiketrains for a single trial.

    params:
        spiketrain: a list of numpy arrays containing the timings of spiking events
        ax: matplotlib axes object on which to plot
        t_min: if specified, spikes before this limit will be discarded
        t_max: if specified, spikes after this point will be discarded
        kwargs: additional key-word arguments will be passed into matplotlib.pyplot.plot
    returns:
        matloblib axes object
    """

    try:
        (_ for _ in spiketrain[0])  # iterable check
        if len(spiketrain) == 1:
            raise TypeError
    except TypeError:
        raise TypeError(
            f"spike_time must be an iterable containing at least array of spiketimes\n"
            f"Passed spiketimes: {spiketrain}"
        )

    if ax is None:
        _, ax = plt.subplots()
    if t_min:
        spiketrain = [
            np.array(list(filter(lambda x: x > t_min, spikes))) for spikes in spiketrain
        ]
    if t_max:
        spiketrain = [
            np.array(list(filter(lambda x: x < t_max, spikes))) for spikes in spiketrain
        ]
    if isinstance(spiketrain, list):
        for i, spikes in enumerate(spiketrain):
            ax = raster_single_neuron(spikes, ax=ax, y_data_ind=i, **kwargs)
    else:
        raise ValueError("must pass in a list of spike times")

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return ax


def psth_raster():
    # TODO
    raise NotImplementedError()

