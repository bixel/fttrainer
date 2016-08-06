from sklearn.cross_validation import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import numpy as np


def bootstrap_calibrate_prob(labels, weights, probs, n_calibrations=30,
                             threshold=0., symmetrize=False):
    """
    Bootstrap isotonic calibration (borrowed from tata-antares/tagging_LHCb):
     * randomly divide data into train-test
     * on train isotonic is fitted and applyed to test
     * on test using calibrated probs p(B+) D2 and auc are calculated

    :param probs: probabilities, numpy.array of shape [n_samples]
    :param labels: numpy.array of shape [n_samples] with labels
    :param weights: numpy.array of shape [n_samples]
    :param threshold: float, to set labels 0/1j
    :param symmetrize: bool, do symmetric calibration, ex. for B+, B-

    :return: D2 array and auc array
    """
    aucs = []
    D2_array = []
    labels = (labels > threshold) * 1

    for _ in range(n_calibrations):
        (train_probs, test_probs,
         train_labels, test_labels,
         train_weights, test_weights) = train_test_split(
            probs, labels, weights, train_size=0.5)
        iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        if symmetrize:
            iso_reg.fit(np.r_[train_probs, 1-train_probs],
                          np.r_[train_labels > 0, train_labels <= 0],
                          np.r_[train_weights, train_weights])
        else:
            iso_reg.fit(train_probs, train_labels, train_weights)

        probs_calib = iso_reg.transform(test_probs)
        alpha = (1 - 2 * probs_calib) ** 2
        aucs.append(roc_auc_score(test_labels, test_probs,
                                  sample_weight=test_weights))
        D2_array.append(np.average(alpha, weights=test_weights))
    return np.array(D2_array), np.array(aucs)
