import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def waveform_peaks_by_neuron(
    df: pd.core.frame.DataFrame,
    neuron_col: str = "neuron_id",
    index_col: str = "waveform_index",
    value_col: str = "waveform_value",
    range_around: int = 70,
    sigma: float = 1,
    diff_threshold: float = 25,
):
    """
    Given a dataframe containing timepoints of average waveforms from multiple neurons,
    calculates the spike initation, minimum and after spike hyper polerisation 
    for each waveform. 

    params:
        neuron_col: column label for column containing neuron ids
        value_col: column label for column containing waveform values
        index_col: column label for column containing waveform index
        diff_col: column label for column containing the differentiated values (col.diff())
        range_aroud: hyperparameter controling maximum range around peak to search for
                     initialisation and ahp peaks
        sigma: hyper parameter controling gaussian smoothing during peak finding 
        diff_threshold: hyperparameter controling minimum slope required for the minimum peak.
                        inscrease to require larger slope.
    
    returns:
        - df containing columns: {neuron_id, peak_name, peak_idx, peak_value}
          peak_idx refers to the index of the peak on the average waveform.
    """
    return (
        df.groupby(neuron_col)
        .apply(
            _get_waveform_peaks_by_neuron,
            index_col=index_col,
            value_col=value_col,
            range_around=range_around,
            sigma=sigma,
            diff_threshold=diff_threshold,
        )
        .reset_index()
        .drop("level_1", axis=1)
    )


def _get_waveform_peaks_by_neuron(
    df: pd.core.frame.DataFrame,
    index_col: str = "waveform_index",
    value_col: str = "waveform_value",
    range_around: int = 70,
    sigma: float = 1,
    diff_threshold: float = 25,
):
    """
    Given a group by object derived from a dataframe containing timepoints of
    average waveforms, calculates the spike initation, minimum and 
    after spike hyper polerisation for each waveform

    params
        value_col: column label for column containing waveform values
        index_col: column label for column containing waveform index
        diff_col: column label for column containing the differentiated values (col.diff())
        range_aroud: hyperparameter controling maximum range around peak to search for
                     initialisation and ahp peaks
        sigma: hyper parameter controling gaussian smoothing during peak finding 
        diff_threshold: hyperparameter controling minimum slope required for the minimum peak.
                        inscrease to require larger slope. 
    """
    return (
        df.sort_values(by=["waveform_index"])
        .pipe(lambda x: x.assign(filt=gaussian_filter1d(x[value_col], sigma=sigma)))
        .pipe(lambda x: x.assign(diff_=x["filt"].diff()))
        .pipe(
            lambda x: _find_waveform_peaks(
                x,
                value_col="filt",
                index_col=index_col,
                range_around=range_around,
                diff_col="diff_",
                diff_threshold=diff_threshold,
            )
        )
    )


def _find_waveform_peaks(
    df: pd.core.frame.DataFrame,
    value_col: str = "filt",
    index_col: str = "waveform_index",
    diff_col: str = "diff_",
    range_around: int = 70,
    diff_threshold: float = 25,
):
    """
    Given a dataframe containing timepoints of one average waveform, 
    calculates the spike initation, minimum and after spike hyper polerisation

    params
        value_col: column label for column containing waveform values
        index_col: column label for column containing waveform index
        diff_col: column label for column containing the differentiated values (col.diff())
        range_aroud: hyperparameter controling maximum range around peak to search for
                     initialisation and ahp peaks
        diff_threshold: hyperparameter controling minimum slope required for the minimum peak.
                        inscrease to require larger slope. 
    """

    min_val = df[value_col].min()
    min_idx = df[df[value_col] == min_val][index_col].values[0]

    try:
        # TODO implement warning
        before_val = (
            df[
                (df[index_col] > (min_idx - range_around))
                & (df[index_col] < min_idx)
                & (
                    np.absolute(df[diff_col])
                    > (diff_threshold * np.absolute(np.nanmedian(df[diff_col])))
                )
            ]
        )[value_col].max()

        before_idx = df[df[value_col] == before_val][index_col].values[0]

    except IndexError:
        return pd.DataFrame(
            {
                "peak_name": ["initiation", "minimum", "ahp"],
                "peak_idx": [np.nan, np.nan, np.nan],
                "peak_value": [np.nan, np.nan, np.nan],
            }
        )

    after_val = df[
        (df[index_col] < (min_idx + range_around)) & (df[index_col] > (min_idx))
    ][value_col].max()
    after_idx = df[df[value_col] == after_val][index_col].values[0]

    return pd.DataFrame(
        {
            "peak_name": ["initiation", "minimum", "ahp"],
            "peak_idx": [before_idx, min_idx, after_idx],
            "peak_value": [before_val, min_val, after_val],
        }
    )

