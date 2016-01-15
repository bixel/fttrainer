"""
Utility functions from original repo github.com/tata-antares/tagging_LHCb
"""

import numpy
import pandas
from collections import OrderedDict

from sklearn import clone
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from matplotlib import pyplot as plt
from rep.utils import train_test_split, train_test_split_group, Flattener
from scipy.special import logit, expit
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve


def union(*arrays):
    return numpy.concatenate(arrays)


def get_events_statistics(data, id_column='event_id'):
    """
    :return: dict with 'Events' - number of events and 'tracks' - number of samples
    """
    return {'Events': len(numpy.unique(data[id_column])), 'tracks': len(data)}


def get_events_number(data, id_column='event_id'):
    """
    :return: number of B events
    """
    _, data_ids = numpy.unique(data[id_column], return_inverse=True)
    weights = numpy.bincount(data_ids, weights=data.N_sig_sw) / numpy.bincount(data_ids)
    return numpy.sum(weights)

def get_N_B_events():
    '''
    :return: number of B decays (sum of sWeight in initial root file) 
    '''
    N_B_decays = 7.42867714256286621e+05
    return N_B_decays

    
def plot_flattened_probs(probs, labels, weights, label=1, check_input=True):
    """
    Prepares transformation, which turns predicted probabilities to uniform in [0, 1] distribution.
    
    :param probs: probabilities, numpy.array of shape [n_samples, 2]
    :param labels: numpy.array of shape [n_samples] with labels (0 and 1)
    :param weights: numpy.array of shape [n_samples]
    :param label: int, predictions of this class will be turned to uniform.
    
    :return: flattener
    """
    if check_input:
        probs, labels, weights = numpy.array(probs), numpy.array(labels), numpy.array(weights)
        assert probs.shape[1] == 2
        assert numpy.in1d(labels, [0, 1]).all()
    
    signal_probs = probs[:, 1]
    flattener = Flattener(signal_probs[labels == label], weights[labels == label])
    flat_probs = flattener(signal_probs)
    
    plt.hist(flat_probs[labels == 1], bins=100, normed=True, histtype='step', 
             weights=weights[labels == 1], label='same sign')
    plt.hist(flat_probs[labels == 0], bins=100, normed=True, histtype='step', 
             weights=weights[labels == 0], label='opposite sign')
    plt.xlabel('predictions')
    plt.legend(loc='upper center')
    plt.show()
    return flattener


def bootstrap_calibrate_prob(labels, weights, probs, n_calibrations=30, group_column=None, threshold=0., symmetrize=False):
    """
    Bootstrap isotonic calibration: 
     * randomly divide data into train-test
     * on train isotonic is fitted and applyed to test
     * on test using calibrated probs p(B+) D2 and auc are calculated 
    
    :param probs: probabilities, numpy.array of shape [n_samples]
    :param labels: numpy.array of shape [n_samples] with labels 
    :param weights: numpy.array of shape [n_samples]
    :param threshold: float, to set labels 0/1 
    :param symmetrize: bool, do symmetric calibration, ex. for B+, B-
    
    :return: D2 array and auc array
    """
    aucs = []
    D2_array = []
    labels = (labels > threshold) * 1
    
    for _ in range(n_calibrations):
        if group_column is not None:
            train_probs, test_probs, train_labels, test_labels, train_weights, test_weights = train_test_split_group(
                group_column, probs, labels, weights, train_size=0.5)
        else:
            train_probs, test_probs, train_labels, test_labels, train_weights, test_weights = train_test_split(
                probs, labels, weights, train_size=0.5)
        iso_est = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        if symmetrize:
            iso_est.fit(numpy.r_[train_probs, 1-train_probs], 
                        numpy.r_[train_labels > 0, train_labels <= 0],
                        numpy.r_[train_weights, train_weights])
        else:
            iso_est.fit(train_probs, train_labels, train_weights)
            
        probs_calib = iso_est.transform(test_probs)
        alpha = (1 - 2 * probs_calib) ** 2
        aucs.append(roc_auc_score(test_labels, test_probs, sample_weight=test_weights))
        D2_array.append(numpy.average(alpha, weights=test_weights))
    return D2_array, aucs


def predict_by_estimator(estimator, datasets):
    '''
    Predict data by classifier
    Important note: this also works correctly if classifier is FoldingClassifier and one of dataframes is his training data.
    
    :param estimator: REP classifier, already trained model.
    :param datasets: list of pandas.DataFrames to predict.
        
    :return: data, probabilities
    '''     
    data = pandas.concat(datasets)    
    # predicting each DataFrame separately to preserve FoldingClassifier
    probs = numpy.concatenate([estimator.predict_proba(dataset)[:, 1] for dataset in datasets])
    return data, probs


def result_table(tagging_efficiency, tagging_efficiency_delta, D2, auc, name='model'):
    """
    Represents results of tagging in a nice table.
    
    :param tagging_efficiency: float, which part of samples will be tagged
    :param tagging_efficiency_delta: standard error of efficiency
    :param D2: D^2, average value ((p(B+) - 0.5)*2)^2 for sample
    :param name: str, name of model
    :param auc: full auc, calculated with untag events (probs are set 0.5) with B+/B- labels
    
    :return: pandas.DataFrame with only one row, describing result_table
    
    Use pandas.concat to get table with results of different methods.
    """
    result = OrderedDict()
    result['name'] = name
    result['$\epsilon_{tag}, \%$'] = [tagging_efficiency * 100.]
    result['$\Delta \epsilon_{tag}, \%$'] = [tagging_efficiency_delta * 100.]
    result['$D^2$'] = [numpy.mean(D2)]
    result['$\Delta D^2$'] = [numpy.std(D2)]
    epsilon = numpy.mean(D2) * tagging_efficiency * 100.
    result['$\epsilon, \%$'] = [epsilon]
    relative_D2_error = numpy.std(D2) / numpy.mean(D2)
    relative_eff_error = tagging_efficiency_delta / tagging_efficiency
    relative_epsilon_error = numpy.sqrt(relative_D2_error ** 2 + relative_eff_error ** 2) 
    result['$\Delta \epsilon, \%$'] = [relative_epsilon_error * epsilon]
    result['AUC, with untag'] = [numpy.mean(auc) * 100]
    result['$\Delta$ AUC, with untag'] = [numpy.std(auc) * 100]
    return pandas.DataFrame(result)


def calibrate_probs(labels, weights, probs, logistic=False, random_state=11, threshold=0., return_calibrator=False, symmetrize=False):
    """
    Calibrate output to probabilities using 2-folding to calibrate all data
    
    :param probs: probabilities, numpy.array of shape [n_samples]
    :param labels: numpy.array of shape [n_samples] with labels 
    :param weights: numpy.array of shape [n_samples]
    :param threshold: float, to set labels 0/1 
    :param logistic: bool, use logistic or isotonic regression
    :param symmetrize: bool, do symmetric calibration, ex. for B+, B-
    
    :return: calibrated probabilities
    """
    labels = (labels > threshold) * 1
    ind = numpy.arange(len(probs))
    ind_1, ind_2 = train_test_split(ind, random_state=random_state, train_size=0.5)
    
    calibrator = LogisticRegression(C=100) if logistic else IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    est_calib_1, est_calib_2 = clone(calibrator), clone(calibrator)
    probs_1 = probs[ind_1]
    probs_2 = probs[ind_2]
    
    if logistic:
        probs_1 = numpy.clip(probs_1, 0.001, 0.999)
        probs_2 = numpy.clip(probs_2, 0.001, 0.999)
        probs_1 = logit(probs_1)[:, numpy.newaxis]
        probs_2 = logit(probs_2)[:, numpy.newaxis]
        if symmetrize:
            est_calib_1.fit(numpy.r_[probs_1, 1-probs_1], 
                            numpy.r_[labels[ind_1] > 0, labels[ind_1] <= 0])
            est_calib_2.fit(numpy.r_[probs_2, 1-probs_2], 
                            numpy.r_[labels[ind_2] > 0, labels[ind_2] <= 0])
        else:
            est_calib_1.fit(probs_1, labels[ind_1])
            est_calib_2.fit(probs_2, labels[ind_2])
    else:
        if symmetrize:
            est_calib_1.fit(numpy.r_[probs_1, 1-probs_1], 
                            numpy.r_[labels[ind_1] > 0, labels[ind_1] <= 0],
                            numpy.r_[weights[ind_1], weights[ind_1]])
            est_calib_2.fit(numpy.r_[probs_2, 1-probs_2], 
                            numpy.r_[labels[ind_2] > 0, labels[ind_2] <= 0],
                            numpy.r_[weights[ind_2], weights[ind_2]])
        else:
            est_calib_1.fit(probs_1, labels[ind_1], weights[ind_1])
            est_calib_2.fit(probs_2, labels[ind_2], weights[ind_2])
        
    calibrated_probs = numpy.zeros(len(probs))
    if logistic:
        calibrated_probs[ind_1] = est_calib_2.predict_proba(probs_1)[:, 1]
        calibrated_probs[ind_2] = est_calib_1.predict_proba(probs_2)[:, 1]
    else:
        calibrated_probs[ind_1] = est_calib_2.transform(probs_1)
        calibrated_probs[ind_2] = est_calib_1.transform(probs_2)
    if return_calibrator:
        return calibrated_probs, (est_calib_1, est_calib_2)
    else:
        return calibrated_probs


def calculate_auc_with_and_without_untag_events(Bsign, Bprobs, Bweights):
    """
    Calculate AUC score for data and AUC full score for data and untag data (p(B+) for untag data is set to 0.5)
    
    :param Bprobs: p(B+) probabilities, numpy.array of shape [n_samples]
    :param Bsign: numpy.array of shape [n_samples] with labels {-1, 1}
    :param Bweights: numpy.array of shape [n_samples]
    
    :return: auc, full auc
    """
    N_B_not_passed = get_N_B_events() - sum(Bweights)
    Bsign_not_passed = [-1, 1]
    Bprobs_not_passed = [0.5] * 2
    Bweights_not_passed = [N_B_not_passed / 2.] * 2
    
    auc_full = roc_auc_score(union(Bsign, Bsign_not_passed), union(Bprobs, Bprobs_not_passed),
                             sample_weight=union(Bweights, Bweights_not_passed))
    auc = roc_auc_score(Bsign, Bprobs, sample_weight=Bweights)
    return auc, auc_full


def compute_B_prob_using_part_prob(data, probs, weight_column='N_sig_sw', event_id_column='event_id', signB_column='signB',
                                   sign_part_column='signTrack', normed_signs=False):
    """
    Compute p(B+) using probs for parts of event (tracks/vertices).
    
    :param data: pandas.DataFrame, data
    :param probs: probabilities for parts of events, numpy.array of shape [n_samples]
    :param weight_column: column for weights in data
    :param event_id_column: column for event id in data
    :param signB_column: column for event B sign in data
    :param sign_part_column: column for part sign in data
    
    :return: B sign array, B weight array, B+ prob array, B event id
    """
    result_event_id, data_ids = numpy.unique(data[event_id_column].values, return_inverse=True)
    log_probs = numpy.log(probs) - numpy.log(1 - probs)
    sign_weights = numpy.ones(len(log_probs))
    if normed_signs:
        for sign in [-1, 1]:
            maskB = (data[signB_column].values == sign)
            maskPart = (data[sign_part_column].values == 1)
            sign_weights[maskB * maskPart] *= sum(maskB * (~maskPart)) * 1. /  sum(maskB * maskPart)
    log_probs *= sign_weights * data[sign_part_column].values
    result_logprob = numpy.bincount(data_ids, weights=log_probs)
    # simply reconstructing original
    result_label = numpy.bincount(data_ids, weights=data[signB_column].values) / numpy.bincount(data_ids)
    result_weight = numpy.bincount(data_ids, weights=data[weight_column]) / numpy.bincount(data_ids)
    return result_label, result_weight, expit(result_logprob), result_event_id


def get_B_data_for_given_part(estimator, datasets, logistic=True, sign_part_column='signTrack', part_name='track',
                              random_state=11, normed_signs=False):
    """
    Predict probabilities for event parts, calibrate it and compute B data.
    Return B data for given part of event:tracks/vertices.
    
    :param estimator: REP classifier, already trained model.
    :param datasets: list of pandas.DataFrames to predict.
    :param logistic: bool, use logistic or isotonic regression for part (track/vertex) probabilities calibration
    :param sign_part_column: column for part sign in data
    :param part_name: part data name for plots 
    
    :return: B sign, weight, p(B+), event id and full auc (with untag events) 
    """
    # Calibration p(track/vertex same sign|B)
    data_calib, part_probs = predict_by_estimator(estimator, datasets)
    part_probs_calib = calibrate_probs(data_calib.label.values, data_calib.N_sig_sw.values, part_probs, 
                                       logistic=logistic, random_state=random_state)
    plt.figure(figsize=[18, 5])
    plt.subplot(1,3,1)
    plt.hist(part_probs[data_calib.label.values == 0], bins=60, normed=True, alpha=0.3, label='os')
    plt.hist(part_probs[data_calib.label.values == 1], bins=60, normed=True, alpha=0.3, label='ss')
    plt.legend(), plt.title('{} probs'.format(part_name))
    
    plt.subplot(1,3,2)
    plt.hist(part_probs_calib[data_calib.label.values == 0], bins=60, normed=True, alpha=0.3, label='os')
    plt.hist(part_probs_calib[data_calib.label.values == 1], bins=60, normed=True, alpha=0.3, label='ss')
    plt.legend(), plt.title('{} probs calibrated'.format(part_name))
        
    all_events = get_events_statistics(data_calib)['Events']
    
    # Compute p(B+)
    Bsign, Bweight, Bprob, Bevent = compute_B_prob_using_part_prob(data_calib, part_probs_calib, 
                                                                   sign_part_column=sign_part_column, normed_signs=normed_signs)
    Bprob[~numpy.isfinite(Bprob)] = 0.5
    Bprob[numpy.isnan(Bprob)] = 0.5
    
    plt.subplot(1,3,3)
    plt.hist(Bprob[numpy.array(Bsign) == -1], bins=60, normed=True, alpha=0.3, label='$B^-$')
    plt.hist(Bprob[numpy.array(Bsign) == 1], bins=60, normed=True, alpha=0.3, label='$B^+$')
    plt.legend(), plt.title('B probs'), plt.show()
    assert all_events == len(Bprob), '{}, {}'.format(all_events, Bprob)
    
    auc, auc_full = calculate_auc_with_and_without_untag_events(Bsign, Bprob, Bweight)
    print 'AUC for tagged:', auc, 'AUC with untag:', auc_full
    return Bsign, Bweight, Bprob, Bevent, auc_full


def get_result_with_bootstrap_for_given_part(tagging_efficiency, tagging_efficiency_delta, estimator,
                                             datasets, name, logistic=True, n_calibrations=30,
                                             sign_part_column='signTrack', part_name='track',
                                             random_state=11, normed_signs=False):
    """
    Predict probabilities for event parts, calibrate it, compute B data and estimate with bootstrap (calibration p(B+)) D2
    
    :param tagging_efficiency: float, which part of samples will be tagged
    :param tagging_efficiency_delta: standard error of efficiency
    :param estimator: REP classifier, already trained model.
    :param datasets: list of pandas.DataFrames to predict.
    :param name: str, name of model
    :param logistic: bool, use logistic or isotonic regression for part (track/vertex) probabilities calibration
    :param sign_part_column: column for part sign in data
    :param part_name: part data name for plots 
    
    :return: pandas.DataFrame with only one row, describing result_table
    """
    Bsign, Bweight, Bprob, Bevent, auc_full = get_B_data_for_given_part(estimator, datasets, logistic=logistic, 
                                                                        sign_part_column=sign_part_column, 
                                                                        part_name=part_name, random_state=random_state,
                                                                        normed_signs=normed_signs)    
    # Compute p(B+) calibrated with bootstrap
    D2, aucs = bootstrap_calibrate_prob(Bsign, Bweight, Bprob, n_calibrations=30)
    print 'mean AUC after calibration:', numpy.mean(aucs), numpy.var(aucs)
    return result_table(tagging_efficiency, tagging_efficiency_delta, D2, auc_full, name)


def prepare_B_data_for_given_part(estimator, datasets, logistic=True, sign_part_column='signTrack', part_name='track', 
                                  random_state=11, normed_signs=False):
    """
    Prepare B data for event parts (track/vetex) for further combination of track-based and vertex-based taggers:
    predict probabilities for event parts, calibrate it, compute B data and p(B+) / (1 - p(B+)) (see formula in description) 
    
    :param estimator: REP classifier, already trained model.
    :param datasets: list of pandas.DataFrames to predict.
    :param name: str, name of model
    :param logistic: bool, use logistic or isotonic regression for part (track/vertex) probabilities calibration
    :param sign_part_column: column for part sign in data
    :param part_name: part data name for plots 
    
    :return: pandas.DataFrame with keys: `event_id` - B id, `Bweight` - B weight, `{part_name}_relation_prob` p(B+) / (1 - p(B+)) for given part, `Bsign` - sign B
    """
    
    Bsign, Bweight, Bprob, Bevent, auc_full = get_B_data_for_given_part(estimator, datasets, logistic=logistic, 
                                                                        sign_part_column=sign_part_column, 
                                                                        part_name=part_name, random_state=random_state,
                                                                        normed_signs=normed_signs)    
    # Roc curve
    fpr, tpr, _ = roc_curve(Bsign, Bprob, sample_weight=Bweight)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.ylim(0, 1), plt.xlim(0, 1), plt.show()
    Bdata_prepared = pandas.DataFrame({'event_id': Bevent, 
                                       'Bweight': Bweight, 
                                       '{}_relation_prob'.format(part_name): Bprob / (1. - Bprob),
                                       'Bsign': Bsign})
    return Bdata_prepared


def compute_mistag(Bprobs, Bsign, Bweight, chosen, uniform=True, bins=None, label=""):
    """
    Check mistag calibration (plot mistag vs true mistag in bins)
    
    :param Bprobs: p(B+) probabilities, numpy.array of shape [n_samples]
    :param Bsign: numpy.array of shape [n_samples] with labels {-1, 1}
    :param Bweights: numpy.array of shape [n_samples]
    :param chosen: condition to select B events (B+ or B- only)
    :param uniform: bool, uniform bins or percentile in the other case
    :params bins: bins
    :param label: label on the plot
    
    """
    if uniform:
        bins = bins
    else:
        bins = numpy.percentile(numpy.minimum(Bprobs, 1 - Bprobs), bins)

    prob = Bprobs[chosen]
    sign = Bsign[chosen]
    weight = Bweight[chosen]
    p_mistag = numpy.minimum(prob, 1 - prob)
    tag = numpy.where(prob >= 0.5, 1, -1)
    is_correct = numpy.where(sign * tag > 0, 1, 0)
    
    bins_index = numpy.searchsorted(bins, p_mistag)
    right_tagged = numpy.bincount(bins_index, weights=is_correct * weight)
    wrong_tagged = numpy.bincount(bins_index, weights=(1 - is_correct) * weight)
    p_mistag_true = wrong_tagged / (right_tagged + wrong_tagged)
    
    bins = [0.] + list(bins) + [0.5]
    bins = numpy.array(bins)
    bins_centers = (bins[1:] + bins[:-1]) / 2
    bins_error = (bins[1:] - bins[:-1]) / 2
    p_mistag_true_error = numpy.sqrt(wrong_tagged * right_tagged) / (wrong_tagged + right_tagged)**1.5
    plt.errorbar(bins_centers, p_mistag_true, xerr=bins_error, yerr=p_mistag_true_error, fmt='.', label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim(-0.05, 0.55), plt.ylim(-0.05, 0.55)
    plt.grid()
