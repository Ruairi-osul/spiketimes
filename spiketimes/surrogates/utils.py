import numpy as np


def _isi(spiketrain: np.ndarray):
    "returns the inter-spike-intervals of a spiketrain"
    return np.diff(np.sort(spiketrain))
