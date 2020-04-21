import numpy as np
import pandas as pd


def align_to(
    to_be_aligned: np.ndarray, to_align_to: np.ndarray, no_beyond: bool = False
):
    """
    Align to_be_aligned to to_align_to.
    For each element in to_be aligned, find the closest element in to_align_to with a lower value.
    Returns np.nan for elements in to_be_aligned smaller than the smallest element in to_be_aligned
    Optionally specify no_beyond. If specified, sets all elements in to_be_aligned larger than to_align_to
    equal to np.nan

    Args:
        to_be_aligned: an np.ndarray to align
        to_align_to: an np.ndarray of events to align to
        no_beyond: if True, returns np.nan for each event in to_be_aligned
                   occuring after the last event in to_align_to
    
    Returns:
        a np.ndarray of to_be_aligned aligned to to_align_to
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


def negative_align(to_be_aligned, to_align_to, no_before=False):
    """
    Negatively align each element in to_be_aligned to to_align_to.
    For each element in to_be_aligned, the latency between it and the closest
    larger element in to_aligned_to is returned.

    Optionally return nan for elements in to_be_aligned occuring before the
    first element in to_align_to
    
    Args:
        to_be_aligned: an np.ndarray to align
        to_align_to: an np.ndarray of events to align to
        no_before: if True, returns np.nan for each event in to_be_aligned
                   occuring before the first event in to_align_to
    
    Returns:
        a np.ndarray of to_be_aligned aligned to to_align_to 
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
    Aligns one array to another. Elements will be negativly aligned if they
    occur at or less than t_before

    Args:
        to_be_aligned: an np.ndarray containing data to be aligned
        to_align_to: a np.ndarray containing data to align to
        t_before: events occuring t_before or less before an event will be
                  negatively aligned to that event. Should be positive.
        max_latency: latencies above this threshold will be returned as nan
        drop: whether to return only non nan element of 

    Returns:
        a numpy array of to_be_aligned aligned to to_align_to
    """
    # TODO: compare to approach used by df compare pos to neg and maybe switch
    postive_latencies = align_to(to_be_aligned, to_align_to, no_beyond=False)

    if t_before is not None:
        negative_latencies = negative_align(to_be_aligned, to_align_to, no_before=False)
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
