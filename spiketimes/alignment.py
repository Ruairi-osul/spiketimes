import numpy as np
import pandas as pd
from .binning import which_bin


def _align_to(
    to_be_aligned: np.ndarray, to_align_to: np.ndarray, no_beyond: bool = False
):
    """
    Align one array to another. Only allows to next smallest event.

    Args:
        to_be_aligned: A numpy array to align
        to_align_to: A numpy to align to (events)
        no_beyond: If True, returns NaN for elements occuring after the maximum element in to_align_to
    Returns:
        A numpy array of aligned data
    """

    if isinstance(to_be_aligned, pd.core.series.Series):
        to_be_aligned = to_be_aligned.values
    if isinstance(to_align_to, pd.core.series.Series):
        to_align_to = to_align_to.values

    _to_be_aligned_isiter = False
    _to_align_to_isiter = False
    try:
        [x for x in to_be_aligned]
        _to_be_aligned_isiter = True
    except TypeError:
        pass
    try:
        [x for x in to_align_to]
        _to_align_to_isiter = True
    except TypeError:
        pass
    if not _to_align_to_isiter and _to_be_aligned_isiter:
        raise TypeError(
            "Must not pass two objects of length one.\n"
            "At least argument must be an iterable"
        )
    if not isinstance(to_be_aligned, np.ndarray) or not isinstance(
        to_align_to, np.ndarray
    ):
        raise TypeError("Both arrays must be numpy arrays")

    if not len(to_align_to.shape) == 1 and not len(to_be_aligned.shape) == 1:
        raise ValueError("Must Pass in flat numpy arrays. Try your_array.flatten()")

    idx = np.searchsorted(to_align_to, to_be_aligned)
    aligned_data = (to_be_aligned - to_align_to[idx - 1]).astype(np.float)
    aligned_data[aligned_data < 0] = np.nan

    if no_beyond:
        aligned_data[to_be_aligned > np.max(to_align_to)] = np.nan
    return aligned_data


def _negative_align(to_be_aligned, to_align_to, no_before=False):
    """
    Align one array to another. Algins to next largest event.

    Args:
        to_be_aligned: A numpy array to align
        to_align_to: A numpy to align to (events)
        no_beyond: If True, returns NaN for elements occuring after the maximum element in to_align_to
    Returns:
        A numpy array of aligned data
    """

    if isinstance(to_be_aligned, pd.core.series.Series):
        to_be_aligned = to_be_aligned.values
    if isinstance(to_align_to, pd.core.series.Series):
        to_align_to = to_align_to.values

    _to_be_aligned_isiter = False
    _to_align_to_isiter = False
    try:
        [x for x in to_be_aligned]
        _to_be_aligned_isiter = True
    except TypeError:
        pass
    try:
        [x for x in to_align_to]
        _to_align_to_isiter = True
    except TypeError:
        pass
    if not _to_align_to_isiter and _to_be_aligned_isiter:
        raise TypeError(
            "Must not pass two objects of length one."
            "At least argument must be an iterable"
        )

    if not isinstance(to_be_aligned, np.ndarray) or not isinstance(
        to_align_to, np.ndarray
    ):
        raise TypeError("Both arrays must be numpy arrays")

    # needs a dummy value to work. This appended value is not aligned to
    to_align_to = np.concatenate([to_align_to, np.array([np.max(to_align_to) + 100])])
    max_idx = len(to_align_to) - 1
    idx = np.searchsorted(to_align_to, to_be_aligned).astype(np.int)
    idx[idx < max_idx] += 1
    aligned_data = (to_be_aligned - to_align_to[idx - 1]).astype(np.float)
    aligned_data[aligned_data > 0] = np.nan
    if no_before:
        aligned_data[to_be_aligned < np.min(to_align_to)] = np.nan
    return aligned_data


def align_around(
    to_be_aligned: np.ndarray,
    to_align_to: np.ndarray,
    t_before: float = None,
    max_latency: float = None,
    drop=False,
):
    """
    Align one array to another.

    Useful for aligning data to events. Default behaviour is to align to closest smaller
    event. If t_before is specified

    Args:
        to_be_aligned: A numpy array to align
        to_align_to: A numpy to align to (events)
        t_before: The time window before each aligning event.
        max_latency: Maximum aligned latency. Latencies above this threshold will be returned as NaN
        drop: Whether to drop NaN elements of the aligned array
    Returns:
        A numpy array of aligned data
    """
    postive_latencies = _align_to(to_be_aligned, to_align_to, no_beyond=False)

    if t_before is not None:
        negative_latencies = _negative_align(
            to_be_aligned, to_align_to, no_before=False
        )
        latencies = np.where(
            (negative_latencies >= (t_before * -1)),
            negative_latencies,
            postive_latencies,
        )
    else:
        latencies = postive_latencies

    if max_latency:
        latencies[latencies > max_latency] = np.nan

    if drop:
        latencies = latencies[np.logical_not(np.isnan(latencies))]
    return latencies


def split_by_trial(
    spiketrain: np.ndarray,
    trial_starts: np.ndarray,
    max_latency: bool = None,
    before: float = None,
):
    """
    Splits a spiketrain into a list of spiketrains by trial.

    Args:
        to_be_aligned: A numpy array to align
        to_align_to: A numpy to align to (events)
        t_before: The time window before each aligning event.
        max_latency: Maximum aligned latency. Latencies above this threshold will be returned as NaN
    Returns:
        A list of numpy arrays of spiketimes in seconds
    """
    idx, val = which_bin(
        spiketrain, trial_starts, max_latency=max_latency, before=before
    )
    return [
        spiketrain[idx == x] - val[idx == x]
        for x in np.unique(idx[np.logical_not(np.isnan(idx))])
    ]
