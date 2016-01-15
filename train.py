#! /usr/bin/env python2
from __future__ import division, print_function

from root_pandas import read_root

from rep.estimators import XGBoostClassifier
from rep.metaml import FoldingClassifier

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

print('Building model...', end='')
sys.stdout.flush()
xgb = XGBoostClassifier(colsample=1.0,
                        eta=0.01,
                        nthreads=4,
                        n_estimators=200,
                        subsample=0.3,
                        max_depth=5)

estimator = FoldingClassifier(xgb,
                              n_folds=2,
                              random_state=11,
                              features=FEATURES)
print(' done.')

print('Start training...', end='')
sys.stdout.flush()
estimator.fit(data, data['label'], data['N_sig_sw'])
print(' done.')
