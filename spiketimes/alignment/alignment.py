import numpy as np


def align_to(to_be_aligned, to_align_to):
    idx = np.searchsorted(to_align_to, to_be_aligned)
    return to_be_aligned - to_align_to[idx - 1]
