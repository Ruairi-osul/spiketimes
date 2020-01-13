import numpy as np


def _isi(spiketrain: np.ndarray):
    return np.diff(np.sort(spiketrain))
