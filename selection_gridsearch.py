from __future__ import print_function

import numpy as np
import pandas as pd
from root_pandas import read_root
import ROOT

import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold
from sklearn.grid_search import RandomizedSearchCV
from xgboost import XGBClassifier
from scripts.metrics import tagging_power_score
from uncertainties import ufloat

from itertools import islice
from tqdm import tqdm
import argparse
import datetime
import json


def get_event_number(df, weight_column='SigYield_sw'):
    """ Use weighted sums
    """
    # max, min, mean, first should give the same values here
    return np.sum(df.groupby('event_id')[weight_column].first())


class CutBasedXGBClassifier(XGBClassifier):
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective="reg:linear",
                 nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0,
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
                 ):
        self.cut_parameters = ['P', 'PT', 'phiDistance', 'MuonPIDIsMuon',
                               'IsSignalDaughter', 'TRCHI2DOF', 'TRGHP',
                               'PROBNNmu', 'PROBNNpi', 'PROBNNe', 'PROBNNk',
                               'PROBNNp', 'IPPUs', 'RecVertexIPs',
                               ]
        for cp in self.cut_parameters:
            setattr(self, '{}_cut'.format(cp), locals()['{}_cut'.format(cp)])
            setattr(self, '{}_column'.format(cp), locals()['{}_column'.format(cp)])
        self.mvaFeatures = mvaFeatures
        self.only_max_pt = only_max_pt
        self.event_identifier_column = event_identifier_column
        self.fit_status_ = True
        super(CutBasedXGBClassifier, self).__init__(max_depth=max_depth, learning_rate=learning_rate,
                                                    n_estimators=n_estimators, silent=silent, objective=objective,
                                                    nthread=nthread, gamma=gamma, min_child_weight=min_child_weight,
                                                    max_delta_step=max_delta_step,
                                                    subsample=subsample, colsample_bytree=colsample_bytree,
                                                    colsample_bylevel=colsample_bylevel,
                                                    reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                                    scale_pos_weight=scale_pos_weight,
                                                    base_score=base_score, seed=seed, missing=None)

    def select(self, X, y=None):
        print('Applying selection')
        len_before = get_event_number(X)
        selection = ((X[self.P_column] > self.P_cut)
                     & (X[self.PT_column] > self.PT_cut)
                     & (X[self.phiDistance_column] > self.phiDistance_cut)
                     & (X[self.MuonPIDIsMuon_column] == self.MuonPIDIsMuon_cut)
                     & (X[self.IsSignalDaughter_column] == self.IsSignalDaughter_cut)
                     & (X[self.TRCHI2DOF_column] < self.TRCHI2DOF_cut)
                     & (X[self.TRGHP_column] < self.TRGHP_cut)
                     & (X[self.PROBNNmu_column] > self.PROBNNmu_cut)
                     & (X[self.PROBNNpi_column] < self.PROBNNpi_cut)
                     & (X[self.PROBNNe_column] < self.PROBNNe_cut)
                     & (X[self.PROBNNk_column] < self.PROBNNk_cut)
                     & (X[self.PROBNNp_column] < self.PROBNNp_cut)
                     & (X[self.IPPUs_column] > self.IPPUs_cut)
                     & (X[self.RecVertexIPs_column] > self.RecVertexIPs_cut)
                    )
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
            if  cutname in kwargs and kwargs[cutname] is not None:
                setattr(self, cutname, kwargs.pop(cutname))
        for other_param in ['mvaFeatures',
                            'only_max_pt',
                            'event_identifier_column']:
            if other_param in kwargs and kwargs[other_param] is not None:
                setattr(self, other_param, kwargs.pop(other_param))
        super(CutBasedXGBClassifier, self).set_params(**kwargs)
        return self

    def fit(self, X, y, eval_set=None, **kwargs):
        if eval_set is not None:
            eval_set = [self.select(X_, y_) for X_, y_ in eval_set]
        X_, y_ = self.select(X, y)
        del(X)
        del(y)
        return super(CutBasedXGBClassifier, self).fit(X_, y_,
                                                      eval_set=eval_set,
                                                      **kwargs)

    def predict_proba(self, data, **kwargs):
        print('Predicting probas')
        d_ = self.select(data)
        del(data)
        return (super(CutBasedXGBClassifier, self)
                .predict_proba(d_, **kwargs))

    def score(self, X, y, sample_weight=None):
        print('Calculating tagging power')
        probas = self.predict_proba(X)[:,1]
        del(X)
        sc = tagging_power_score(probas, efficiency=self.efficiency_,
                                 sample_weight=sample_weight)
        return sc


class SelectionKFold(KFold):
    def __init__(self, y, n_folds=3, shuffle=False, random_state=None):
        self.event_ids = y.event_id
        self.unique_events = self.event_ids.unique()
        self.raw_indices = np.arange(len(y))
        (super(SelectionKFold, self)
                .__init__(len(self.unique_events), n_folds=n_folds,
                          shuffle=shuffle, random_state=random_state))

    def __iter__(self):
        for train_ind, test_ind in super(SelectionKFold, self).__iter__():
            print('Yielding split')
            yield (self.raw_indices[(self.event_ids
                                    .isin(self.unique_events[train_ind])
                                    .values)],
                   self.raw_indices[(self.event_ids
                                    .isin(self.unique_events[test_ind])
                                    .values)])

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--slices', type=int, default=None)
    parser.add_argument('-n', '--n-folds', type=int, default=2)
    parser.add_argument('-x', '--xgboost-jobs', type=int, default=1)
    parser.add_argument('-c', '--cv-jobs', type=int, default=1)
    parser.add_argument('-i', '--iter', type=int, default=100)
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
    filenames = [
        data_dir + 'Bu2JpsiK_mu-k-e-TrainingTuple_2011_MD_sweighted_kheinick.root',
        data_dir + 'Bu2JpsiK_mu-k-e-TrainingTuple_2011_MU_sweighted_kheinick.root',
        data_dir + 'Bu2JpsiK_mu-k-e-TrainingTuple_2012_MD_sweighted_kheinick.root',
        data_dir + 'Bu2JpsiK_mu-k-e-TrainingTuple_2012_MU_sweighted_kheinick.root',
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

    skf = SelectionKFold(df, n_folds=args.n_folds)
    cbxgb = CutBasedXGBClassifier(mvaFeatures=classic_MVA_features,
                                  max_depth=5,
                                  n_estimators=300,
                                  seed=1)

    grid_searcher = RandomizedSearchCV(
            CutBasedXGBClassifier(nthread=args.xgboost_jobs,
                                  mvaFeatures=classic_MVA_features,
                                  n_estimators=300),
            {
                'P_cut': np.linspace(2, 5, 10),
                'PT_cut': np.linspace(0, 2, 10),
                'phiDistance_cut': np.linspace(0, 0.5, 10),
                'MuonPIDIsMuon_cut': [0, 1],
                'TRGHP_cut': np.linspace(0, 0.6, 10),
                'IPPUs_cut': np.linspace(1, 4, 10),
                'n_estimators': [200, 300, 400],
            },
            n_iter=args.iter,
            error_score=0,
            verbose=1,
            cv=skf,
            pre_dispatch='n_jobs',
            n_jobs=args.cv_jobs,
            refit=False)

    grid_searcher.fit(df, df.target)
    best_index = np.nanargmax([sc[1] for sc in grid_searcher.grid_scores_])
    best_score = grid_searcher.grid_scores_[best_index]
    print('tagging power {}% with params\n{}'
          .format(100 * ufloat(np.mean(best_score[2]),
                  np.std(best_score[2])),
                  best_score[0]))
    best_params = best_score[0]
    with open('search-{:%Y%m%d-%H%M%S}.json'
              .format(datetime.datetime.now()), 'w') as f:
        json.dump({'settings': vars(args),
                   'scores': [{'parameters': nt.parameters,
                               'mean_validation_score': nt.mean_validation_score,
                               'cv_validation_scores': nt.cv_validation_scores.tolist(),
                              } for nt in grid_searcher.grid_scores_],
                  }, f, indent=4)


if __name__ == '__main__':
    main()
