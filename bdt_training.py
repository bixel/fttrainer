#! /usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

import numpy as np
import pandas as pd

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from root_pandas import read_root

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

import json
import argparse
import sys
from collections import OrderedDict
from tqdm import tqdm
from itertools import islice
from textwrap import dedent
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

from scripts.data_preparation import get_event_number, NSplit
from scripts.calibration import PolynomialLogisticRegression
from scripts.metrics import tagging_power_score, d2_score


def add_features(df):
    df['']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, default=None)
    parser.add_argument('-m', '--max-slices', type=int, default=None)
    parser.add_argument('--bootstrap-folds', type=int, default=None)
    return parser.parse_args()


def parse_config(filename):
    if filename is None:
        return None
    with open(filename, 'r') as f:
        return json.load(f)


def main():
    args = parse_args()
    config = parse_config(args.config_file)
    if config is None:
        print('No configuration file is defined. '
              'Define one with `--config-file`.')
        sys.exit(1)

    # read dataset
    files = config['files']
    if 'filepath' in config:
        files = [config['filepath'] + f for f in files]
    kwargs = config['pandas_kwargs']

    print('Reading ', end='')
    entries = 0
    for f in files:
        rootfile = ROOT.TFile(f)
        tree = rootfile.Get(kwargs['key'])
        entries += tree.GetEntries()
    maxslices = args.max_slices
    chunksize = kwargs['chunksize']
    total = (maxslices
             if maxslices is not None and maxslices < (entries / chunksize)
             else (entries / chunksize))
    print(total * chunksize, 'events.')
    df = pd.concat([
        df for df in tqdm(
            islice(
                read_root(files, flatten=True, **kwargs), maxslices),
            total=total)])

    # rename the tagging particle branches
    df.rename(columns=dict(zip(df.columns,
        [c.replace(config['tagging_particle_prefix'], 'tp').replace('-', '_')
            for c in df.columns])),
        inplace=True)
    df['event_id'] = df.runNumber.apply(str) + '_' + df.eventNumber.apply(str)
    if 'invert_target' in config and config['invert_target']:
        df['target'] = np.sign(df.B_ID) != np.sign(df.tp_ID)
    else:
        df['target'] = np.sign(df.B_ID) == np.sign(df.tp_ID)

    # read features and selections
    try:
        if 'inclusive_mva_features' in config:
            mva_features = ['tp_' + f for f in config['inclusive_mva_features']]
        else:
            mva_features = ['tp_' + f.split(' ')[0] for f in config['selections']]
    except:
        raise ValueError('Tried to parse features for the BDT.'
                         ' Either provide well-formatted `selections` or'
                         ' define a `inclusive_mva_features` set.')

    # build BDT model and train the classifier n_cv x 3 times
    xgb_kwargs = config['xgb_kwargs']
    n_jobs = config['n_jobs']

    bootstrap_scores = []
    bootstrap_d2s = []
    nfold = (args.bootstrap_folds
             if args.bootstrap_folds is not None
             else config['n_cv'])
    print('Starting bootstrapping.')
    pbar = tqdm(total=nfold * 3)
    for _ in range(nfold):
        # yield 3-fold split for CV
        df_sets = [df.iloc[indices] for indices in NSplit(df)]

        cv_scores = []
        for i in range(3):
            df1, df2, df3 = (df_sets[i % 3].copy(),
                             df_sets[(i + 1) % 3].copy(),
                             df_sets[(i + 2) % 3].copy())
            model = XGBClassifier(nthread=n_jobs, **xgb_kwargs)
            sample_weight = (df1.target
                             if 'training_weights' in config
                                and config['training_weights']
                             else None)
            model.fit(df1[mva_features], df1.target,
                      sample_weight=df1.SigYield_sw)

            df2['probas'] = model.predict_proba(df2[mva_features])[:, 1]
            df2.reset_index(inplace=True, drop=True)
            df2_max = df2.iloc[df2.groupby('event_id')['probas'].idxmax()].copy()
            df3['probas'] = model.predict_proba(df3[mva_features])[:, 1]
            df3.reset_index(inplace=True, drop=True)
            df3_max = df3.iloc[df3.groupby('event_id')['probas'].idxmax()].copy()

            # calibrate
            calibrator = PolynomialLogisticRegression(power=4,
                                                      solver='lbfgs',
                                                      n_jobs=n_jobs)
            calibrator.fit(df2_max.probas.reshape(-1, 1), df2_max.target,
                           sample_weight=df2_max.SigYield_sw)

            df3_max['calib_probas'] = calibrator.predict_proba(df3_max.probas)[:, 1]

            score = tagging_power_score(df3_max.calib_probas,
                                        tot_event_number=get_event_number(df3_max),
                                        sample_weight=df3_max.SigYield_sw)
            bootstrap_scores.append(score)
            bootstrap_d2s.append(d2_score(df3_max.calib_probas,
                                          sample_weight=df3_max.SigYield_sw))
            pbar.update(1)

    pbar.close()
    print(dedent("""\
          Final {}-fold bootstrap performance
             D2 = {:<6}%
          ε_eff = {:<6}%""")
          .format(nfold,
                  100 * ufloat(np.mean(bootstrap_d2s),
                               np.std(bootstrap_d2s)),
                  100 * ufloat(np.mean(noms(bootstrap_scores)),
                               np.std(noms(bootstrap_scores)))))


if __name__ == '__main__':
    main()
