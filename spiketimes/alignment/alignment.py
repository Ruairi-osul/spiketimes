import numpy as np


def align_to(to_be_aligned, to_align_to, no_beyond=False):
    """Align to_be_aligned to to_align_to.
    For each element in to_be aligned, find the closest element in to_align_to with a lower value.
    Returns np.nan for elements in to_be_aligned smaller than the smallest element in to_be_aligned
    Optionally specify no_beyond. If specified, sets all elements in to_be_aligned larger than to_align_to
    equal to np.nan
    """

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

    if not len(to_align_to.shape) == 1 and not len(to_be_aligned.shape) == 1:
        raise ValueError("Must Pass in flat numpy arrays. Try your_array.flatten()")

    if no_beyond:
        to_be_aligned = to_be_aligned[to_be_aligned < np.max(to_align_to)]

    idx = np.searchsorted(to_align_to, to_be_aligned)
    aligned_data = (to_be_aligned - to_align_to[idx - 1]).astype(np.float)
    aligned_data[aligned_data < 0] = np.nan
    return aligned_data


def negative_align(to_be_aligned, to_align_to):
    """Align each element in to_be_aligned to the closest larger element
    in to_align_to"""
    max_idx = len(to_align_to) - 1
    idx = np.searchsorted(to_align_to, to_be_aligned).astype(np.int)
    idx[idx < max_idx] += 1
    aligned_data = (to_be_aligned - to_align_to[idx - 1]).astype(np.float)
    aligned_data[aligned_data > 0] = np.nan
    return aligned_data
