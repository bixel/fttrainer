from __future__ import division

import numpy as np
from root_pandas import read_root


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


def tagging_power_score(df, config, total_event_number=None,
                        selected_event_number=None, efficiency=None,
                        etas='etas'):
    """ Compute per event tagging power with selection efficiency

    Parameters
    ----------
    df : pandas DataFrame
        a pandas DataFrame, expected to contain the columns
            - eta_column (default 'etas', optional, see etas)
            - 'target'
            - a weight branch (defined in the config object)
    config : dictionary
        a configuration object expected to contain the keys
            - all keys required for get_event_number
            - 'sorting_feature' used to select a single tagging particle
    etas : str or array, optional, default='etas'
        name of the column used as mistag prediction or list of eta values
    total_event_number : float, optional, default=None
        provide the total number of events in the base dataset for df.
        If this is none, the number will be extracted from the provided config
        file

    Returns
    -------
    score : float
        tagging power
    """
    if 'runNumber' not in df.index.names:
        df.set_index(['runNumber', 'eventNumber', '__array_index'], inplace=True)

    if type(etas) == str:
        etas = df[etas]

    sorting_feature = config['sorting_feature']
    grouped = df.groupby(['runNumber', 'eventNumber'], sort=False)
    idxmax = grouped[sorting_feature].idxmax()
    max_df = df.loc[idxmax]
    efficiency = (efficiency
                  or ((selected_event_number or max_df.SigYield_sw.sum())
                      / (total_event_number or get_event_number(config))
                      )
                  )
    return ((max_df.SigYield_sw * (1 - 2 * etas)**2).sum()
            / max_df.SigYield_sw.sum() * efficiency)
