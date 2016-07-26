import numpy as np


def avg_dilution(y_true, y_score, threshhold=0.5):
    wrong_tags = y_true != (y_score > threshhold)
    omega = np.sum(wrong_tags) / len(y_true)
    return 1 - 2*omega


def tagging_power(efficiency, dilution):
    return efficiency * dilution ** 2


def tagging_power_score(y_true, y_score, sample_weight=None, efficiency=None):
    if not efficiency:
        efficiency = 1

    dilution = avg_dilution(y_true, y_score)
    return tagging_power(efficiency, dilution)
