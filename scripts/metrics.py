import numpy as np


def tagging_power_score(y_true, y_score, sample_size, sample_weight=None):
    """Compute per event tagging power
    """
    if sample_weight is None:
        sample_weight = 1

    mistag_estimate = 1 - y_score
    D2s = (1 - 2 * mistag_estimate)**2
    return 1 / sample_size * np.sum(sample_weight * D2s)
