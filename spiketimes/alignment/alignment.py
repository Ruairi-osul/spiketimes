import numpy as np


def align_to(to_be_aligned, to_align_to):
    """Align to_be_aligned to to_align_to i.e. for each element in
    to_be aligned, find the closest element in to_align_to with a lower value (occured before it)
    """
    idx = np.searchsorted(to_align_to, to_be_aligned)
    return to_be_aligned - to_align_to[idx - 1]
