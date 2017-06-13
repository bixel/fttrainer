from __future__ import division

import numpy as np


def get_event_number(config):
    """ Compute the total number of events contained in the base tuples of
    a given configuration.

    Parameters
    ----------
    config : dictionary
        expected to contain the keys
            - 'filepath'
            - 'files'
            - 'pandas_kwargs'
    """
    files = [config['filepath'] + f for f in config['files']]
    df = read_root(files, key=config['pandas_kwargs']['key'],
                   columns=['SigYield_sw', 'nCandidate'])
    return df[df.nCandidate == 0].SigYield_sw.sum()


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
        sample_weight = np.ones_like(y_score)
    D2s = (1 - 2 * y_score)**2

    # This seems to return an unexpected nan value from time to time
    # return np.average(D2s, weights=sample_weight)
    return np.sum(sample_weight * D2s) / np.sum(sample_weight)


def tagging_power_score(y_score, efficiency=None, tot_event_number=None,
                        sample_weight=None):
    """ Compute per event tagging power with selection efficiency

    Parameters
    ----------
    y_score : array-like, shape=(n_samples,)
        omega or p(correct) values

    efficiency : float, optional, default: None
        the selection efficiency

    tot_event_number : float, optional, default: None
        the total number of events (tagged and untagged)

    sample_weight : array-like, shape=(n_samples,), optional, default: None
        Weights. If set to None, all weights will be set to 1

    Returns
    -------
    score : float
        tagging power
    """
    if sample_weight is None:
        sample_weight = np.ones_like(y_score)

    if efficiency is not None and tot_event_number is None:
        return efficiency * d2_score(y_score, sample_weight)
    if tot_event_number is not None and efficiency is None:
        return 1 / tot_event_number * np.sum(sample_weight
                                             * (1 - 2 * y_score)**2)
    else:
        raise("Either efficiency or tot_event_number must be passed!")
