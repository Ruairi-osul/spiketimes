import random
import numpy as np


def _random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def p_adjust(pvalues: np.ndarray, method="Benjamini-Hochberg"):
    """                                                                                                   
    Adjust p values for multiple comparisons using various methods

    params:
        pvalues: numpy array of pvalues from various comparisons
        method: p value correction method. Set of availible methods
                comprise {'Bonferroni', 'Bonferroni-Holm', 'Benjamini-Hochberg'}

    returns:
        numpy array of adjusted pvalues 
    """

    n = pvalues.shape[0]
    new_pvalues = np.empty(n)

    if method == "Bonferroni":
        new_pvalues = n * pvalues

    elif method == "Bonferroni-Holm":
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        for rank, vals in enumerate(values):
            pvalue, i = vals
            new_pvalues[i] = (n - rank) * pvalue

    elif method == "Benjamini-Hochberg":
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = n - i
            pvalue, index = vals
            new_values.append((n / rank) * pvalue)
        for i in range(0, int(n) - 1):
            if new_values[i] < new_values[i + 1]:
                new_values[i + 1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            new_pvalues[index] = new_values[i]

    return new_pvalues


def ppois(q: int, mu: int, tail: str = "two_tailed"):
    """
    Calculates the cumulative of the Poisson distribution
    """
    from scipy.stats import poisson

    if tail == "lower":
        result = poisson.cdf(k=q, mu=mu)
    elif tail == "upper":
        result = 1 - poisson.cdf(k=q, mu=mu)

    if tail == "two_tailed":
        if q < mu:
            result = poisson.cdf(k=q, mu=mu)
        else:
            result = 1 - poisson.cdf(k=q, mu=mu)

    return result


def dpois(x, mu):
    """
    Calculates the density/point estimate of the Poisson distribution
    """
    from scipy.stats import poisson

    result = poisson.pmf(k=x, mu=mu)
    return result
