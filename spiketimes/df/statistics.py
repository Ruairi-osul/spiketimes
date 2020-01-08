import pandas as pd


def mean_firing_rate_ifr_by_neuron(
    df: pd.core.frame.DataFrame, neuron_col: str = "neuron_id", ifr_col: str = "ifr"
):
    """
    Given a dataframe of intantaneous firing rates of many neurons, calculates the 
    mean firing rate of each neuron

    params:
        df: the df containg the data
        neuron_col: label for the column containing neuron ids
        ifr_col: label for the column containing the ifr values

    returns:
        a pd.DataFrame containing with columns {neuron_id, mean_firing_rate} 
    """

    return (
        df.groupby(neuron_col)
        .apply(lambda x: mean_firing_rate_ifr(x, ifr_col=ifr_col))
        .reset_index()
        .rename(columns={0: "mean_firing_rate"})
    )


def mean_firing_rate_ifr(df: pd.core.frame.DataFrame, ifr_col: str = "ifr"):
    """
    Given a dataframe containing a ifr column, calculates the mean 
    """
    return df[ifr_col].mean()
