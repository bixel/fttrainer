#! /usr/bin/env python2
from __future__ import division, print_function

from root_pandas import read_root

import xgboost as xgb

import numpy as np

from uncertainties import ufloat

import sys
import os

from scripts.utils import get_events_statistics

# Some constant definitions
# TODO: handle with configargparse?
E_DATA_PATH = os.path.abspath('./data/nnet_ele.root')
FEATURES = ['mult',
            'partPt',
            'partP',
            'ptB',
            'IPs',
            'partlcs',
            'eOverP',
            'ghostProb',
            'IPPU']
SWEIGHT_THRESHOLD = 0.

print('Reading data...', end='')
sys.stdout.flush()
data = read_root(E_DATA_PATH)
print(' done.')

# prepare dataset
data['label'] = data.iscorrect
data['event_id'] = data.runNum.apply(str) + '_' + data.evtNum.apply(str)

# Check for nonempty dataset
assert not data[FEATURES].empty, 'FEATURE selection wrong, or dataset empty'

print("""    File `{}`
    containing {} tracks and {} events.""".format(
    E_DATA_PATH,
    get_events_statistics(data)['tracks'],
    get_events_statistics(data)['Events'])
)

cut = data.N_sig_sw > SWEIGHT_THRESHOLD

print("""Cutting sweight {}
    {} tracks passing
    {} tracks lost""".format(
    SWEIGHT_THRESHOLD,
    get_events_statistics(data[cut])['tracks'],
    get_events_statistics(data[~cut])['tracks'],
))

# Build the tree, prepare input data, define xgb parameters
print('Building model...', end='')
sys.stdout.flush()

mask = np.random.rand(len(data)) < 0.7

xgb_training_data = xgb.DMatrix(data[cut & mask].loc[:, FEATURES],
                                label=data[cut & mask]['label'],
                                weight=data[cut & mask]['N_sig_sw'],
                                missing=-999.0)
xgb_test_data = xgb.DMatrix(data[cut & ~mask].loc[:, FEATURES],
                            label=data[cut & ~mask]['label'],
                            weight=data[cut & ~mask]['N_sig_sw'],
                            missing=-999.0)
xgb_full_data = xgb.DMatrix(data[cut].loc[:, FEATURES],
                            label=data[cut]['label'],
                            weight=data[cut]['N_sig_sw'],
                            missing=-999.0)

# store the full dataset in xgboost binary format
# TODO: Move this into cpp
xgb_full_data.save_binary("data/nnet_ele.xmat")
evaluation_list = [
    (xgb_test_data, 'eval'),
    (xgb_training_data, 'train'),
]
parameters = {
    'bst:max_depth': 5,
    'bst:eta': 0.01,
    'objective': 'binary:logistic',
    'nthread': 4,
    'eval_metric': 'auc',
    'silent': 1,
}
print(' done.')

# run the actual training
print('Start training...', end='')
sys.stdout.flush()
training_rounds = 200
bst = xgb.train(parameters,
                xgb_training_data,
                training_rounds,
                evaluation_list)
print(' done.')
print('Start cross-validation...', end='')
sys.stdout.flush()
bst.cv = xgb.cv(parameters,
                xgb_full_data,
                nfold=30,
                show_progress=False,
                metrics={'auc', 'error'})
print(' done.')

# print some numbers
max_ind = bst.cv['test-auc-mean'].values.argmax()
last_auc = ufloat(bst.cv['test-auc-mean'].values[max_ind],
                  bst.cv['test-auc-std'].values[max_ind])
print('Max AUC: {} at {}'.format(last_auc, max_ind))

print('Saving model to `models/ele_trained.xgb`...', end='')
bst.save_model('models/ele_trained.xgb')
print(' done.')
