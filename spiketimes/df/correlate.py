import pandas as pd
import numpy as np
from itertools import combinations


def correlate_all_neurons(
    df: pd.core.frame.DataFrame,
    neuron_col: str = "neuron_id",
    binned_spikes_col: str = "spike_count",
):
    frames: list = []
    neurons = df[neuron_col].unique()
    for comb in combinations(neurons, r=2):
        frames.append(
            pd.Series(
                {
                    "neuron_1": comb[0],
                    "neuron_2": comb[1],
                    "pearson_r": np.corrcoef(
                        df[df[neuron_col] == comb[0]][binned_spikes_col],
                        df[df[neuron_col] == comb[1]][binned_spikes_col],
                    )[0, 1],
                }
            )
        )
    return pd.concat(frames, axis=1).transpose()


def correlate_all_neurons_by(
    df: pd.core.frame.DataFrame,
    by_col: str = "session_name",
    neuron_col: str = "neuron_id",
    binned_spikes_col: str = "spike_count",
):
    return (
        df.groupby(by_col)
        .apply(lambda x: correlate_all_neurons(x))
        .reset_index()
        .drop("level_1", axis=1)
    )
