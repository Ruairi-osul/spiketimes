import pandas as pd
import numpy as np
import multiprocessing
from itertools import combinations, product
import spiketimes.correlate
from ..utils import p_adjust


def spike_count_correlation(
    df: pd.core.frame.DataFrame,
    binsize: int,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    min_firing_rate: float = None,
    t_start: float = None,
    t_stop: float = None,
    use_multiprocessing: bool = False,
    max_cores: int = None,
):
    """
    Calculate pearson's correlation coefficient of spike counts between all pairs of spiketrains in a dataframe.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        binsize: The size of the time bin in seconds
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column containing spiketrain identifiers
        t_start: start point for first time bin.
        t_stop: end point for the last time bin.
        use_multiprocessing: Whether to use multiple cores to compute cross correlation.
                             Useful for large numbers of spiketrains
        max_cores: If using multiprocessing, specifies the maximum number of cores to use.
                   Defaults to max available
    Returns:
        A pandas DataFrame with columns {spiketrain_1, spiketrain_2, R_spike_count}
    """
    if t_start is None:
        t_start = np.min(df[spiketimes_col])
    if t_stop is None:
        t_stop = np.max(df[spiketimes_col])

    spiketrain_ids = df[spiketrain_col].unique()

    spiketrain_combs = list(combinations(spiketrain_ids, r=2))
    args = [
        [
            df[df[spiketrain_col] == spiketrain_1][spiketimes_col].values,
            df[df[spiketrain_col] == spiketrain_2][spiketimes_col].values,
            binsize,
            min_firing_rate,
            t_start,
            t_stop,
        ]
        for spiketrain_1, spiketrain_2 in spiketrain_combs
    ]
    if not use_multiprocessing:
        res = []
        for i, arg_set in enumerate(args):
            spiketrain_1, spiketrain_2 = spiketrain_combs[i]
            res.append(
                pd.DataFrame(
                    {
                        "spiketrain_1": spiketrain_1,
                        "spiketrain_2": spiketrain_2,
                        "R_spike_count": spiketimes.correlate.spike_count_correlation(
                            *arg_set
                        ),
                    },
                    index=[0],
                )
            )
        return pd.concat(res, axis=0).reset_index(drop=True)
    if max_cores:
        with multiprocessing.Pool(max_cores) as p:
            res = p.starmap(spiketimes.correlate.spike_count_correlation, args)
    else:
        with multiprocessing.Pool() as p:
            res = p.starmap(spiketimes.correlate.spike_count_correlation, args)
    stiketrain_1 = [st1 for st1, _ in spiketrain_combs]
    spiketrain_2 = [st2 for _, st2 in spiketrain_combs]
    return pd.DataFrame(
        {
            "spiketrain_1": stiketrain_1,
            "spiketrain_2": spiketrain_2,
            "R_spike_count": res,
        }
    )


def spike_count_correlation_between(
    df: pd.core.frame.DataFrame,
    binsize: int,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    group_col: str = "group",
    min_firing_rate: float = None,
    t_start: float = None,
    t_stop: float = None,
    use_multiprocessing: bool = False,
    max_cores: int = None,
):
    """
    Calculate spike count correlation between all pairs of spiketrains of different groups.

    For example: correlate all pairs of fast-spiking and slow-spining neurons.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        binsize: The size of the time bin in seconds
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column containing spiketrain identifiers
        group_col: The label of the column containing group identifiers
        t_start: start point for first time bin.
        t_stop: end point for the last time bin.
        use_multiprocessing: Whether to use multiple cores to compute cross correlation.
                             Useful for large numbers of spiketrains
        max_cores: If using multiprocessing, specifies the maximum number of cores to use.
                   Defaults to max available
    Returns:
        A pandas DataFrame with columns {spiketrain_1, spiketrain_2, R_spike_count}
    """
    if t_start is None:
        t_start = np.min(df[spiketimes_col])
    if t_stop is None:
        t_stop = np.max(df[spiketimes_col])
    groups = df[group_col].unique()
    frames: list = []
    for group_1, group_2 in combinations(groups, r=2):
        # get spiketrians
        spiketrains_group_1 = df.loc[df[group_col] == group_1][spiketrain_col].unique()
        spiketrains_group_2 = df.loc[df[group_col] == group_2][spiketrain_col].unique()
        spiketrain_combs = list(product(spiketrains_group_1, spiketrains_group_2))
        # get args
        args = [
            [
                df[df[spiketrain_col] == spiketrain_1][spiketimes_col].values,
                df[df[spiketrain_col] == spiketrain_2][spiketimes_col].values,
                binsize,
                min_firing_rate,
                t_start,
                t_stop,
            ]
            for spiketrain_1, spiketrain_2 in spiketrain_combs
        ]
        # get results for this group combination
        if not use_multiprocessing:
            res = []
            for i, arg_set in enumerate(args):
                spiketrain_1, spiketrain_2 = spiketrain_combs[i]
                res.append(
                    pd.DataFrame(
                        {
                            "spiketrain_1": spiketrain_1,
                            "spiketrain_2": spiketrain_2,
                            "R_spike_count": spiketimes.correlate.spike_count_correlation(
                                *arg_set
                            ),
                        },
                        index=[0],
                    )
                )
            frames.append(
                pd.concat(res, axis=0)
                .assign(**{"group_1": group_1, "group_2": group_2})
                .reset_index(drop=True)
            )
        else:
            if max_cores:
                with multiprocessing.Pool(max_cores) as p:
                    res = p.starmap(spiketimes.correlate.spike_count_correlation, args)
            else:
                with multiprocessing.Pool() as p:
                    res = p.starmap(spiketimes.correlate.spike_count_correlation, args)
            stiketrain_1 = [st1 for st1, _ in spiketrain_combs]
            spiketrain_2 = [st2 for _, st2 in spiketrain_combs]
            frames.append(
                pd.DataFrame(
                    {
                        "spiketrain_1": stiketrain_1,
                        "spiketrain_2": spiketrain_2,
                        "group_1": group_1,
                        "group_2": group_2,
                        "R_spike_count": res,
                    }
                )
            )
    return pd.concat(frames, axis=0)


def spike_count_correlation_test(
    df: pd.core.frame.DataFrame,
    binsize: int,
    n_boot: int = 500,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    min_firing_rate: float = None,
    t_start: float = None,
    t_stop: float = None,
    tail: str = "two_tailed",
    adjust_p: bool = True,
    p_adjust_method: str = "Benjamini-Hochberg",
    use_multiprocessing: bool = False,
    max_cores: int = None,
):
    """
    Calculate spike count correlation between all pairs of spiketrains of different groups.

    For example: correlate all pairs of fast-spiking and slow-spining neurons.
    Multiprocessing recommeded when computing on large datasets.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        binsize: The size of the time bin in seconds
        n_boot: The number of bootstrap replicates to create.
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column containing spiketrain identifiers
        group_col: The label of the column containing group identifiers
        t_start: The start point for first time bin.
        t_stop: The end point for the last time bin.
        tail: Tail for hypothesis test {"two_tailed", "upper", "lower"}. Two tailed reccomended
        adjust_p: Whether to adjust p-values for multiple comparisons.
        p_adjust_method: If adjusting p-values, specified which method to use {Benjamini-Hochberg', 'Bonferroni',
                        'Bonferroni-Holm'}
        use_multiprocessing: Whether to use multiple cores to compute cross correlation.
                             Useful for large numbers of spiketrains
        max_cores: If using multiprocessing, specifies the maximum number of cores to use.
                   Defaults to max available
    Returns:
        A pandas DataFrame with columns {spiketrain_1, spiketrain_2, R_spike_count}
    """
    if t_start is None:
        t_start = np.min(df[spiketimes_col])
    if t_stop is None:
        t_stop = np.max(df[spiketimes_col])

    spiketrain_ids = df[spiketrain_col].unique()

    spiketrain_combs = list(combinations(spiketrain_ids, r=2))
    args = [
        [
            df[df[spiketrain_col] == spiketrain_1][spiketimes_col].values,
            df[df[spiketrain_col] == spiketrain_2][spiketimes_col].values,
            binsize,
            n_boot,
            min_firing_rate,
            t_start,
            t_stop,
            tail,
        ]
        for spiketrain_1, spiketrain_2 in spiketrain_combs
    ]
    if not use_multiprocessing:
        res = []
        for i, arg_set in enumerate(args):
            spiketrain_1, spiketrain_2 = spiketrain_combs[i]
            r, p = spiketimes.correlate.spike_count_correlation_test(*arg_set)
            res.append(
                pd.DataFrame(
                    {
                        "spiketrain_1": spiketrain_1,
                        "spiketrain_2": spiketrain_2,
                        "R_spike_count": r,
                        "p": p,
                    },
                    index=[0],
                )
            )
        df = pd.concat(res, axis=0).reset_index(drop=True)
    if max_cores:
        with multiprocessing.Pool(max_cores) as p:
            res = p.starmap(spiketimes.correlate.spike_count_correlation_test, args)
    else:
        with multiprocessing.Pool() as p:
            res = p.starmap(spiketimes.correlate.spike_count_correlation_test, args)
    stiketrain_1 = [st1 for st1, _ in spiketrain_combs]
    spiketrain_2 = [st2 for _, st2 in spiketrain_combs]
    r = [r for r, _ in res]
    p = [p for _, p in res]
    df = pd.DataFrame(
        {
            "spiketrain_1": stiketrain_1,
            "spiketrain_2": spiketrain_2,
            "R_spike_count": r,
            "p": p,
        }
    )
    if adjust_p:
        df["p"] = p_adjust(df["p"].values, method=p_adjust_method)
    return df


def spike_count_correlation_between_test(
    df: pd.core.frame.DataFrame,
    binsize: int,
    n_boot: int = 500,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    group_col: str = "group",
    min_firing_rate: float = None,
    t_start: float = None,
    t_stop: float = None,
    tail: str = "two_tailed",
    adjust_p: bool = True,
    p_adjust_method: str = "Benjamini-Hochberg",
    use_multiprocessing: bool = False,
    max_cores: int = None,
):
    """
    Calculate spike count correlation between all pairs of spiketrains of different groups.
    Also test significance using a bootstrap procedure.

    For example: correlate all pairs of fast-spiking and slow-spining neurons.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        binsize: The size of the time bin in seconds
        n_boot: The number of bootstrap replicates to create.
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column containing spiketrain identifiers
        group_col: The label of the column containing group identifiers
        t_start: The start point for first time bin.
        t_stop: The end point for the last time bin.
        tail: Tail for hypothesis test {"two_tailed", "upper", "lower"}. Two tailed reccomended
        adjust_p: Whether to adjust p-values for multiple comparisons.
        p_adjust_method: If adjusting p-values, specified which method to use {Benjamini-Hochberg', 'Bonferroni',
                        'Bonferroni-Holm'}
        use_multiprocessing: Whether to use multiple cores to compute cross correlation.
                             Useful for large numbers of spiketrains
        max_cores: If using multiprocessing, specifies the maximum number of cores to use.
                   Defaults to max available
    Returns:
        A pandas DataFrame with columns {spiketrain_1, spiketrain_2, R_spike_count}
    """
    if t_start is None:
        t_start = np.min(df[spiketimes_col])
    if t_stop is None:
        t_stop = np.max(df[spiketimes_col])
    groups = df[group_col].unique()
    frames: list = []
    for group_1, group_2 in combinations(groups, r=2):
        # get spiketrians
        spiketrains_group_1 = df.loc[df[group_col] == group_1][spiketrain_col].unique()
        spiketrains_group_2 = df.loc[df[group_col] == group_2][spiketrain_col].unique()
        spiketrain_combs = list(product(spiketrains_group_1, spiketrains_group_2))
        # get args
        args = [
            [
                df[df[spiketrain_col] == spiketrain_1][spiketimes_col].values,
                df[df[spiketrain_col] == spiketrain_2][spiketimes_col].values,
                binsize,
                n_boot,
                min_firing_rate,
                t_start,
                t_stop,
                tail,
            ]
            for spiketrain_1, spiketrain_2 in spiketrain_combs
        ]
        # get results for this group combination
        if not use_multiprocessing:
            res = []
            for i, arg_set in enumerate(args):
                spiketrain_1, spiketrain_2 = spiketrain_combs[i]
                r, p = spiketimes.correlate.spike_count_correlation_test(*arg_set)
                res.append(
                    pd.DataFrame(
                        {
                            "spiketrain_1": spiketrain_1,
                            "spiketrain_2": spiketrain_2,
                            "R_spike_count": r,
                            "p": p,
                        },
                        index=[0],
                    )
                )
            frames.append(
                pd.concat(res, axis=0)
                .assign(**{"group_1": group_1, "group_2": group_2})
                .reset_index(drop=True)
            )
        else:
            if max_cores:
                with multiprocessing.Pool(max_cores) as p:
                    res = p.starmap(
                        spiketimes.correlate.spike_count_correlation_test, args
                    )
            else:
                with multiprocessing.Pool() as p:
                    res = p.starmap(
                        spiketimes.correlate.spike_count_correlation_test, args
                    )
            stiketrain_1 = [st1 for st1, _ in spiketrain_combs]
            spiketrain_2 = [st2 for _, st2 in spiketrain_combs]
            r = [r for r, _ in res]
            p = [p for _, p in res]
            frames.append(
                pd.DataFrame(
                    {
                        "spiketrain_1": stiketrain_1,
                        "spiketrain_2": spiketrain_2,
                        "group_1": group_1,
                        "group_2": group_2,
                        "R_spike_count": r,
                        "p": p,
                    }
                )
            )
    df = pd.concat(frames, axis=0)
    if adjust_p:
        df["p"] = p_adjust(df["p"].values, method=p_adjust_method)
    return df


def auto_corr(
    df: pd.core.frame.DataFrame,
    binsize: int = 0.01,
    num_lags: int = 100,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    t_start: float = None,
    t_stop: float = None,
):
    """
    Calculate the autocorrelation function for each spiketrain in a DataFrame.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        binsize: The size of the time bin in seconds
        num_lags: The number of lags forward and backwards around lag 0 to return
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column containing spiketrain identifiers
        t_start: Minimum timepoint
        t_stop: Maximum timepoint
    Returns:
        A pandas DataFrame with columns {spiketrain, time_bin, autocorrelation}
    """
    return (
        df.groupby(spiketrain_col)
        .apply(
            lambda x: spiketimes.correlate.auto_corr(
                x[spiketimes_col], binsize=binsize, num_lags=num_lags, as_df=True
            )
        )
        .reset_index()
        .drop("level_1", axis=1)
    )


def cross_corr(
    df: pd.core.frame.DataFrame,
    binsize: float = 0.01,
    num_lags: int = 100,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    t_start: float = None,
    t_stop: float = None,
    use_multiprocessing: bool = False,
    max_cores: int = None,
):
    """
    Calculate crosscorrelation between each combination of spiketrains in a DataFrame.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        binsize: The size of the time bin in seconds
        num_lags: The number of lags forward and backwards around lag 0 to return
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column containing spiketrain identifiers
        t_start: Minimum timepoint
        t_stop: Maximum timepoint
        use_multiprocessing: Whether to use multiple cores to compute cross correlation.
                             Useful for large numbers of spiketrains
        max_cores: If using multiprocessing, specifies the maximum number of cores to use.
                   Defaults to max available
    Returns:
        A pandas DataFrame with columns {spiketrain_1, spiketrain_2, time_bin, crosscorrelation}
    """
    spiketrain_ids = df[spiketrain_col].unique()

    spiketrain_combs = list(combinations(spiketrain_ids, r=2))
    args = [
        [
            df[df[spiketrain_col] == spiketrain_1][spiketimes_col].values,
            df[df[spiketrain_col] == spiketrain_2][spiketimes_col].values,
            binsize,
            num_lags,
            True,
            t_start,
            t_stop,
        ]
        for spiketrain_1, spiketrain_2 in spiketrain_combs
    ]
    if not use_multiprocessing:
        res = []
        for i, arg_set in enumerate(args):
            spiketrain_1, spiketrain_2 = spiketrain_combs[i]
            res.append(
                spiketimes.correlate.cross_corr(*arg_set).assign(
                    **{"spiketrain_1": spiketrain_1, "spiketrain_2": spiketrain_2}
                )
            )
    else:
        if max_cores:
            with multiprocessing.Pool(max_cores) as p:
                res = p.starmap(spiketimes.correlate.cross_corr, args)
        else:
            with multiprocessing.Pool() as p:
                res = p.starmap(spiketimes.correlate.cross_corr, args)

        for spiketrain_comb, r in zip(spiketrain_combs, res):
            r["spiketrain_1"] = spiketrain_comb[0]
            r["spiketrain_2"] = spiketrain_comb[1]
    df = pd.concat(res)
    return df[["spiketrain_1", "spiketrain_2", "time_bin", "crosscorrelation"]]


def cross_corr_test(
    df: pd.core.frame.DataFrame,
    binsize: float = 0.01,
    num_lags: int = 100,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    t_start: int = None,
    t_stop: int = None,
    tail: str = "two_tailed",
    adjust_p: bool = True,
    p_adjust_method: str = "Benjamini-Hochberg",
    use_multiprocessing: bool = False,
    max_cores: int = None,
):
    """
    Calculate spike count correlation between all pairs of spiketrains. Also test significance of crosscorrelation.

    Significance test performed by comparing observed crosscorrelation to expected cross correlation of
    poisson spiketrains.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        binsize: The size of the time bin in seconds
        num_lags: The number of lags forward and backwards around lag 0 to return
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column containing spiketrain identifiers
        group_col: The label of the column containing group identifiers
        t_start: Minimum timepoint
        t_stop: Maximum timepoint
        tail: Tail for hypothesis test {"two_tailed", "upper", "lower"}. Two tailed reccomended
        adjust_p: Whether to adjust p-values for multiple comparisons.
        p_adjust_method: If adjusting p-values, specified which method to use {Benjamini-Hochberg', 'Bonferroni',
                        'Bonferroni-Holm'}
        use_multiprocessing: Whether to use multiple cores to compute cross correlation.
                             Useful for large numbers of spiketrains
        max_cores: If using multiprocessing, specifies the maximum number of cores to use.
                   Defaults to max available
    Returns:
        A pandas DataFrame with columns {spiketrain_1, spiketrain_2, group_1, group_2, time_bin, crosscorrelation, p}
    """
    spiketrain_ids = df[spiketrain_col].unique()

    spiketrain_combs = list(combinations(spiketrain_ids, r=2))
    args = [
        [
            df[df[spiketrain_col] == spiketrain_1][spiketimes_col].values,
            df[df[spiketrain_col] == spiketrain_2][spiketimes_col].values,
            binsize,
            num_lags,
            True,  # as_df
            t_start,
            t_stop,
            tail,  # tail
            False,  # adjust_p
            None,  # p_adust_method
        ]
        for spiketrain_1, spiketrain_2 in spiketrain_combs
    ]
    if not use_multiprocessing:
        res = []
        for i, arg_set in enumerate(args):
            spiketrain_1, spiketrain_2 = spiketrain_combs[i]
            res.append(
                spiketimes.correlate.cross_corr_test(*arg_set).assign(
                    **{"spiketrain_1": spiketrain_1, "spiketrain_2": spiketrain_2}
                )
            )
    else:
        if max_cores:
            with multiprocessing.Pool(max_cores) as p:
                res = p.starmap(spiketimes.correlate.cross_corr_test, args)
        else:
            with multiprocessing.Pool() as p:
                res = p.starmap(spiketimes.correlate.cross_corr_test, args)

        for spiketrain_comb, r in zip(spiketrain_combs, res):
            r["spiketrain_1"] = spiketrain_comb[0]
            r["spiketrain_2"] = spiketrain_comb[1]

    df = pd.concat(res, axis=0)

    if adjust_p:
        df["p"] = p_adjust(df["p"].values, method=p_adjust_method)

    return df[["spiketrain_1", "spiketrain_2", "time_bin", "crosscorrelation", "p"]]


def cross_corr_between_test(
    df: pd.core.frame.DataFrame,
    binsize: float = 0.01,
    num_lags: int = 100,
    spiketimes_col: str = "spiketimes",
    spiketrain_col: str = "spiketrain",
    group_col: str = "group",
    t_start: float = None,
    t_stop: float = None,
    tail: str = "two_tailed",
    adjust_p: bool = True,
    p_adjust_method: str = "Benjamini-Hochberg",
    use_multiprocessing: bool = False,
    max_cores: int = None,
):
    """
    Calculate spike count correlation between all pairs of spiketrains of different groups.

    For example: correlate all pairs of fast-spiking and slow-spining neurons.

    Args:
        df: A pandas DataFrame containing spiketimes indexed by spiketrain
        binsize: The size of the time bin in seconds
        num_lags: The number of lags forward and backwards around lag 0 to return
        spiketimes_col: The label of the column containing spiketimes
        spiketrain_col: The label of the column containing spiketrain identifiers
        t_start: Minimum timepoint
        t_stop: Maximum timepoint
        tail: Tail for hypothesis test {"two_tailed", "upper", "lower"}. Two tailed reccomended
        adjust_p: Whether to adjust p-values for multiple comparisons.
        p_adjust_method: If adjusting p-values, specified which method to use {Benjamini-Hochberg', 'Bonferroni',
                        'Bonferroni-Holm'}
        use_multiprocessing: Whether to use multiple cores to compute cross correlation.
                             Useful for large numbers of spiketrains
        max_cores: If using multiprocessing, specifies the maximum number of cores to use.
                   Defaults to max available
    Returns:
        A pandas DataFrame with columns {spiketrain_1, spiketrain_2, group_1, group_2, time_bin, crosscorrelation, p}
    """
    if t_start is None:
        t_start = np.min(df[spiketimes_col])
    if t_stop is None:
        t_stop = np.max(df[spiketimes_col])
    groups = df[group_col].unique()
    frames: list = []
    for group_1, group_2 in combinations(groups, r=2):
        # get spiketrians
        spiketrains_group_1 = df.loc[df[group_col] == group_1][spiketrain_col].unique()
        spiketrains_group_2 = df.loc[df[group_col] == group_2][spiketrain_col].unique()
        spiketrain_combs = list(product(spiketrains_group_1, spiketrains_group_2))
        # get args
        args = [
            [
                df[df[spiketrain_col] == spiketrain_1][spiketimes_col].values,
                df[df[spiketrain_col] == spiketrain_2][spiketimes_col].values,
                binsize,
                num_lags,
                True,
                t_start,
                t_stop,
                tail,
                False,
                None,
            ]
            for spiketrain_1, spiketrain_2 in spiketrain_combs
        ]
        # get results for this group combination
        if not use_multiprocessing:
            res = []
            for i, arg_set in enumerate(args):
                spiketrain_1, spiketrain_2 = spiketrain_combs[i]
                res.append(
                    spiketimes.correlate.cross_corr_test(*arg_set).assign(
                        **{"spiketrain_1": spiketrain_1, "spiketrain_2": spiketrain_2}
                    )
                )
        else:
            if max_cores:
                with multiprocessing.Pool(max_cores) as p:
                    res = p.starmap(spiketimes.correlate.cross_corr_test, args)
            else:
                with multiprocessing.Pool() as p:
                    res = p.starmap(spiketimes.correlate.cross_corr_test, args)

            for spiketrain_comb, r in zip(spiketrain_combs, res):
                r["spiketrain_1"] = spiketrain_comb[0]
                r["spiketrain_2"] = spiketrain_comb[1]
        frames.append(
            pd.concat(res, axis=0)
            .assign(**{"group_1": group_1, "group_2": group_2})
            .reset_index(drop=True)
        )
    df = pd.concat(frames, axis=0)
    if adjust_p:
        df["p"] = p_adjust(df["p"].values, method=p_adjust_method)
    return df
