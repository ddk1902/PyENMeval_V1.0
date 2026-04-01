# pyenmeval/metrics_aicc.py

import numpy as np


def log_likelihood(y_pred):
    """
    Approximate log-likelihood for MaxEnt (presence-only).

    Parameters
    ----------
    y_pred : array-like
        Predicted probabilities for presence points

    Returns
    -------
    float
    """

    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

    return np.sum(np.log(y_pred))


def compute_aicc(logLik, k, n):
    """
    Compute AICc (corrected Akaike Information Criterion)

    Parameters
    ----------
    logLik : float
    k : int
        number of parameters
    n : int
        number of samples

    Returns
    -------
    float
    """

    if np.isnan(k) or k <= 0:
        return np.nan

    aic = -2 * logLik + 2 * k

    if (n - k - 1) <= 0:
        return np.nan

    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)

    return aicc