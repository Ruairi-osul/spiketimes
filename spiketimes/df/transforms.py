from ..statistics import ifr


def df_ifr(df, spiketime_col="spike_time_samples", fs=1, t_start=None, t_stop=None):
    if t_stop is None:
        t_stop = df[spiketime_col].max()
    if t_start is None:
        t_start = df[spiketime_col].min()
    return df.sort_values(spiketime_col, axis="rows").pipe(
        lambda x: ifr(x[spiketime_col], fs=fs, t_start=t_start, t_stop=t_stop)
    )


def ifr_by_neuron(
    df,
    neuron_col,
    spiketime_col="spike_time_samples",
    ifr_fs=1,
    t_start=None,
    t_stop=None,
):
    if t_stop is None:
        t_stop = df[spiketime_col].max()
    if t_start is None:
        t_start = df[spiketime_col].min()
    return (
        df.groupby(neuron_col)
        .apply(
            lambda x: df_ifr(
                x,
                spiketime_col=spiketime_col,
                fs=ifr_fs,
                t_start=t_start,
                t_stop=t_stop,
            )
        )
        .reset_index()
        .drop("level_1", axis=1)
    )
