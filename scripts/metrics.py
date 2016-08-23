import numpy as np


def d2_score(y_score, sample_weight=None):
    """ Compute <D^2> = <(1 - 2*omega)^2> where omega is either the per-event
    mistag estimate or the per event probability of the tag being correct.

    Parameters
    ----------
    y_score : array-like, shape=(n_samples,)
        omega or p(correct) values

    sample_weight : array-like, shape=(n_samples,), optional, default: None
        Weights. If set to None, all weights will be set to 1

    Returns
    -------
    score : float
        D squared
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y_score))
    D2s = (1 - 2 * y_score)**2

    # This seems to return an unexpected nan value from time to time
    # return np.average(D2s, weights=sample_weight)
    return np.sum(sample_weight * D2s) / np.sum(sample_weight)


def tagging_power_score(y_score, efficiency=1, sample_weight=None):
    """ Compute per event tagging power with selection efficiency

    Parameters
    ----------
    y_score : array-like, shape=(n_samples,)
        omega or p(correct) values

    efficiency : float, optional, default: 1
        the selection efficiency

    sample_weight : array-like, shape=(n_samples,), optional, default: None
        Weights. If set to None, all weights will be set to 1

    Returns
    -------
    score : float
        tagging power
    """
    return efficiency * d2_score(y_score, sample_weight)
