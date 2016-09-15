#! /usr/bin/env python

from __future__ import print_function

import numpy as np
import pandas as pd

import ROOT
# disable cmd line parsing before other ROOT deps are loaded
ROOT.PyConfig.IgnoreCommandLineOptions = True
from root_pandas import read_root

import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold, train_test_split
from sklearn.grid_search import RandomizedSearchCV
from xgboost import XGBClassifier
from scripts.metrics import tagging_power_score
from uncertainties import ufloat

from scripts.calibration import PolynomialLogisticRegression
from scripts.data_preparation import NSplit

from collections import OrderedDict
from itertools import islice, product
from tqdm import tqdm
import argparse
import datetime
import json
from os import path, mkdir


def get_event_number(df, weight_column='SigYield_sw'):
    """ Use weighted sums
    """
    # max, min, mean, first should give the same values here
    return np.sum(df.groupby('event_id')[weight_column].first())


class CutBasedXGBClassifier(XGBClassifier):
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective="reg:linear",
                 nthread=1, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None,
                 P_column='tp_partP', P_cut=0,
                 PT_column='tp_partPt', PT_cut=1.1,
                 phiDistance_column='tp_minPhiDistance', phiDistance_cut=0.005,
                 MuonPIDIsMuon_column='tp_MuonPIDIsMuon', MuonPIDIsMuon_cut=1,
                 mvaFeatures=None, only_max_pt=True,
                 event_identifier_column='event_id',
                 IsSignalDaughter_column='tp_IsSignalDaughter',
                 IsSignalDaughter_cut=0,
                 RecVertexIPs_column='tp_ABS_RecVertexIP',
                 RecVertexIPs_cut=0,
                 TRCHI2DOF_column='tp_partlcs', TRCHI2DOF_cut=3,
                 TRGHP_column='tp_ghostProb', TRGHP_cut=0.4,
                 PROBNNmu_column='tp_PIDNNm', PROBNNmu_cut=0.35,
                 PROBNNpi_column='tp_PROBNNpi', PROBNNpi_cut=0.8,
                 PROBNNe_column='tp_PROBNNe', PROBNNe_cut=0.8,
                 PROBNNk_column='tp_PROBNNk', PROBNNk_cut=0.8,
                 PROBNNp_column='tp_PROBNNp', PROBNNp_cut=0.8,
                 IPPUs_column='tp_IPPU', IPPUs_cut=3,
                 polynomial_pow=3,
                 ):
        self.cut_parameters = ['P', 'PT', 'phiDistance', 'MuonPIDIsMuon',
                               'IsSignalDaughter', 'TRCHI2DOF', 'TRGHP',
                               'PROBNNmu', 'PROBNNpi', 'PROBNNe', 'PROBNNk',
                               'PROBNNp', 'IPPUs', 'RecVertexIPs',
                               ]
        for cp in self.cut_parameters:
            setattr(self,
                    '{}_cut'.format(cp),
                    locals()['{}_cut'.format(cp)])
            setattr(self,
                    '{}_column'.format(cp),
                    locals()['{}_column'.format(cp)])
        self.calibrator = PolynomialLogisticRegression(power=polynomial_pow,
                                                       solver='lbfgs',
                                                       n_jobs=nthread)
        self.mvaFeatures = mvaFeatures
        self.only_max_pt = only_max_pt
        self.event_identifier_column = event_identifier_column
        self.fit_successful = False
        (super(CutBasedXGBClassifier, self)
         .__init__(max_depth=max_depth, learning_rate=learning_rate,
                   n_estimators=n_estimators,
                   silent=silent, objective=objective,
                   nthread=nthread, gamma=gamma,
                   min_child_weight=min_child_weight,
                   max_delta_step=max_delta_step,
                   subsample=subsample, colsample_bytree=colsample_bytree,
                   colsample_bylevel=colsample_bylevel,
                   reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                   scale_pos_weight=scale_pos_weight,
                   base_score=base_score, seed=seed, missing=None))

    def select(self, X, y=None):
        len_before = get_event_number(X)
        selection = ((X[self.P_column] > self.P_cut) &
                     (X[self.PT_column] > self.PT_cut) &
                     (X[self.phiDistance_column] > self.phiDistance_cut) &
                     (X[self.MuonPIDIsMuon_column] == self.MuonPIDIsMuon_cut) &
                     (X[self.IsSignalDaughter_column] == self.IsSignalDaughter_cut) &
                     (X[self.TRCHI2DOF_column] < self.TRCHI2DOF_cut) &
                     (X[self.TRGHP_column] < self.TRGHP_cut) &
                     (X[self.PROBNNmu_column] > self.PROBNNmu_cut) &
                     (X[self.PROBNNpi_column] < self.PROBNNpi_cut) &
                     (X[self.PROBNNe_column] < self.PROBNNe_cut) &
                     (X[self.PROBNNk_column] < self.PROBNNk_cut) &
                     (X[self.PROBNNp_column] < self.PROBNNp_cut) &
                     (X[self.IPPUs_column] > self.IPPUs_cut) &
                     (X[self.RecVertexIPs_column] > self.RecVertexIPs_cut))
        X = X[selection]
        if y is not None:
            y = y[selection]

        if self.only_max_pt:
            X.reset_index(drop=True, inplace=True)
            max_pt_indices = (X
                              .groupby(self.event_identifier_column)[self.PT_column]
                              .idxmax())
            X = X.iloc[max_pt_indices]
            if y is not None:
                y.reset_index(drop=True, inplace=True)
                y = y.iloc[max_pt_indices]

        len_after = get_event_number(X)
        self.efficiency_ = len_after / len_before

        if self.mvaFeatures:
            X = X[self.mvaFeatures]

        if y is not None:
            return X, y
        else:
            return X

    def get_params(self, deep=False):
        params = super(CutBasedXGBClassifier, self).get_params(deep=deep)
        for cp in self.cut_parameters:
            params['{}_cut'.format(cp)] = getattr(self,
                                                  '{}_cut'.format(cp))
            params['{}_column'.format(cp)] = getattr(self,
                                                     '{}_column'.format(cp))
        params['mvaFeatures'] = self.mvaFeatures
        params['only_max_pt'] = self.only_max_pt
        params['event_identifier_column'] = self.event_identifier_column
        return params

    def set_params(self, **kwargs):
        for cp in self.cut_parameters:
            cutname = '{}_cut'.format(cp)
            if cutname in kwargs and kwargs[cutname] is not None:
                setattr(self, cutname, kwargs.pop(cutname))
        for other_param in ['mvaFeatures',
                            'only_max_pt',
                            'event_identifier_column']:
            if other_param in kwargs and kwargs[other_param] is not None:
                setattr(self, other_param, kwargs.pop(other_param))
        super(CutBasedXGBClassifier, self).set_params(**kwargs)
        return self

    def fit(self, X, y, X_calib, y_calib, eval_set=None, **kwargs):
        if eval_set is not None:
            raise "Passing an eval_set to xgb ist not yet implemented"
        X, y = self.select(X, y)
        if len(X) < 2:
            return self
        if X_calib is None or y_calib is None:
            X, X_calib, y, y_calib = train_test_split(X, y, test_size=0.5)
        else:
            X_calib, y_calib = self.select(X_calib, y_calib)
        super(CutBasedXGBClassifier, self).fit(X, y, **kwargs)
        probas = super(CutBasedXGBClassifier, self).predict_proba(X_calib)[:, 1]
        self.calibrator.fit(probas.reshape(-1, 1), y_calib)
        self.fit_successful = True
        return self

    def predict_proba(self, data, **kwargs):
        data = self.select(data)
        try:
            uncalibrated = (super(CutBasedXGBClassifier, self)
                            .predict_proba(data, **kwargs))
            return self.calibrator.predict_proba(uncalibrated[:, 1])
        except:
            return np.array([[0.5, 0.5]])

    def score(self, X, y, sample_weight=None):
        probas = self.predict_proba(X)[:, 1]
        sc = tagging_power_score(probas, efficiency=self.efficiency_,
                                 sample_weight=sample_weight)
        return sc


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--slices', type=int, default=None)
    parser.add_argument('-n', '--n-folds', type=int, default=2)
    parser.add_argument('-x', '--xgboost-jobs', type=int, default=1)
    parser.add_argument('-c', '--cv-jobs', type=int, default=1)
    parser.add_argument('-i', '--iter', type=int, default=100)
    parser.add_argument('-y', '--year', type=int, default=2011)
    return parser.parse_args()


def main():
    args = setup_args()

    # style setup
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['font.size'] = 14

    # just define some keyword arguments for read_root in a separate dict
    chunksize = 5000
    data_kwargs = dict(
        key='DecayTree',  # the tree name
        columns=['B_OS_Muon*',  # all branches that should be read
                 'B_ID',
                 'B_PT',
                 'SigYield_sw',
                 'runNumber',
                 'eventNumber',
                 ],
        chunksize=chunksize,  # this will create a generator, yielding subsets
                              # with 'chunksize' of the data
        where='(B_LOKI_MASS_JpsiConstr_NoPVConstr>0)',  # a ROOT where
                                                        # selection, does not
                                                        # work with
                                                        # array-variables
        flatten=True  # will flatten the data in the dimension of the first
                      # given column
    )

    data_dir = '/home/kheinicke/tank/flavourtagging/'
    year = args.year
    filenames = [
        data_dir + 'Bu2JpsiK_mu-k-e-TrainingTuple_' + str(year) + '_MD_sweighted_kheinick.root',
        data_dir + 'Bu2JpsiK_mu-k-e-TrainingTuple_' + str(year) + '_MU_sweighted_kheinick.root',
        data_dir + 'Bu2JpsiK_mu-k-e-TrainingTuple_' + str(year + 1) + '_MD_sweighted_kheinick.root',
        data_dir + 'Bu2JpsiK_mu-k-e-TrainingTuple_' + str(year + 1) + '_MU_sweighted_kheinick.root',
    ]
    chunksize = 5000

    # still have to use plain ROOT to get the number of entries...
    n_entries = 0
    for fn in filenames:
        f = ROOT.TFile(fn)
        t = f.Get('DecayTree')
        n_entries += t.GetEntries()

    # This will read chunks of the data inside a list comprehension and then
    # concat those to a big dataframe note that tqdm is just some boilerplate
    # to generate a progressbar
    if args.slices is None:
        total = n_entries / chunksize
    else:
        total = args.slices
    print('Reading {} chunks of {}-data'.format(total, year))
    df = pd.concat([df for df in tqdm(
        islice(read_root(filenames, **data_kwargs), args.slices),
        total=total)
    ])

    # prepare dataframe
    df['target'] = np.sign(df.B_ID) == np.sign(df.B_OS_Muon_ID)
    df.rename(columns=dict(zip(df.columns, [c.replace('B_OS_Muon', 'tp')
                                            for c in df.columns])),
              inplace=True)
    df['tp_ABS_RecVertexIP'] = np.abs(df.tp_RecVertexIP)
    df['event_id'] = df.runNumber.apply(str) + '_' + df.eventNumber.apply(str)

    # this is the list of BDT variables formerly used
    classic_MVA_features = ['tp_' + c for c in [
        'partP',
        'partPt',
        'IPPU',
        'ghostProb',
        'PIDNNm',
        'ABS_RecVertexIP',
        'mult',
        'ptB',
        'IPs',
    ]]

    # instead of using the sklearn randomized gridsearch, implement one that
    # handles 3-fold validation
    parameter_grid = OrderedDict([
            ('P_cut', np.linspace(2, 5, 31)),
            ('PT_cut', np.linspace(0, 3, 31)),
            ('phiDistance_cut', np.linspace(0, 0.5, 11)),
            ('TRGHP_cut', np.linspace(0, 0.6, 11)),
            ('IPPUs_cut', np.linspace(1, 4, 31)),
        ])

    # produce all possible grid points
    grid = []
    for cut_values in product(*parameter_grid.values()):
        grid.append({p: v for p, v in zip(parameter_grid.keys(), cut_values)})
    np.random.shuffle(grid)

    # iterate the grid points
    scores = []
    print('Starting grid search')
    for classifier_args in tqdm(grid[:args.iter]):
        # yield 3-fold split for this grid point
        df_sets = [df.iloc[indices] for indices in NSplit(df)]

        cv_scores = []
        for i in range(3):
            df1, df2, df3 = df_sets[i % 3], df_sets[(i + 1) % 3], df_sets[(i + 2) % 3]
            model = CutBasedXGBClassifier(nthread=args.xgboost_jobs,
                                          mvaFeatures=classic_MVA_features,
                                          n_estimators=300,
                                          **classifier_args)
            model.fit(df1, df1.target,
                      df2, df2.target)
            if model.fit_successful:
                cv_scores.append(model.score(df3, df3.target))
            else:
                cv_scores.append(0)

        scores.append((np.mean(cv_scores), cv_scores, classifier_args))

    scores = sorted(scores, key=lambda s: s[0])
    print('tagging power {}% with params\n{}'
          .format(100 * ufloat(scores[0][0], np.std(scores[0][1])),
                  scores[0][2]))
    if not path.isdir('build/'):
        mkdir('build/')
        print('build/ directory created')
    output_filename = ('build/search-{:%Y%m%d-%H%M%S}.json'
                       .format(datetime.datetime.now()))
    with open(output_filename, 'w') as f:
        json.dump({'settings': vars(args),
                   'scores': scores,
                   }, f, indent=4)
        print('grid search results have been written to {}'
              .format(output_filename))


if __name__ == '__main__':
    main()
