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
    selection_query = ' and '.join(['tp_' + f for f in config['selections']])
    mva_features = ['tp_' + f for f in config['mva_features']]

    # apply selections
    selected_df = df.query(selection_query)

    # select max pt data
    selected_df.reset_index(drop=True, inplace=True)
    max_df = selected_df.iloc[selected_df
                              .groupby('event_id')[config['sorting_feature']]
                              .idxmax()].copy()
    max_df['probas'] = 0.5
    max_df['calib_probas'] = 0.5
    total_event_number = get_event_number(df)
    total_event_number = ufloat(total_event_number, np.sqrt(total_event_number))
    selected_event_number = get_event_number(max_df)
    selected_event_number = ufloat(selected_event_number, np.sqrt(selected_event_number))
    efficiency = selected_event_number / total_event_number
    avg_omega = np.sum(max_df.SigYield_sw * ~max_df.target) / np.sum(max_df.SigYield_sw)
    avg_omega = ufloat(avg_omega, avg_omega / np.sqrt(selected_event_number.n))
    avg_dilution = (1 - 2 * avg_omega)**2
    avg_tagging_power = efficiency * avg_dilution
    print(dedent("""\
          Average performance values:
          ε_tag = {:<6}%
              ⍵ = {:<6}%
             D2 = {:<6}%
          ε_eff = {:<6}%"""
          .format(100 * efficiency,
                  100 * avg_omega,
                  100 * avg_dilution,
                  100 * avg_tagging_power)))

    # build BDT model and train the classifier n_cv x 3 times
    xgb_kwargs = config['xgb_kwargs']
    n_jobs = config['n_jobs']

    bootstrap_scores = []
    bootstrap_d2s = []
    nfold = config['n_cv']
    print('Starting bootstrapping.')
    pbar = tqdm(total=nfold * 3)
    for _ in range(nfold):
        # yield 3-fold split for CV
        df_sets = [max_df.iloc[indices] for indices in NSplit(max_df)]

        cv_scores = []
        for i in range(3):
            df1, df2, df3 = (df_sets[i % 3].copy(),
                             df_sets[(i + 1) % 3].copy(),
                             df_sets[(i + 2) % 3].copy())
            model = XGBClassifier(nthread=n_jobs, **xgb_kwargs)
            model.fit(df1[mva_features], df1.target,
                      sample_weight=df1.SigYield_sw)

            probas = model.predict_proba(df2[mva_features])[:, 1]
            df2['probas'] = probas
            max_df.loc[df2.index, 'probas'] = probas

            # calibrate
            calibrator = PolynomialLogisticRegression(power=3,
                                                      solver='lbfgs',
                                                      n_jobs=n_jobs)
            calibrator.fit(df2.probas.reshape(-1, 1), df2.target)

            probas = model.predict_proba(df3[mva_features])[:, 1]
            calib_probas = calibrator.predict_proba(probas)[:, 1]
            df3['calib_probas'] = calib_probas
            max_df.loc[df3.index, 'calib_probas'] = calib_probas

            score = tagging_power_score(calib_probas,
                                        efficiency=efficiency,
                                        sample_weight=df3.SigYield_sw)
            bootstrap_scores.append(score)
            bootstrap_d2s.append(d2_score(calib_probas,
                                          sample_weight=df3.SigYield_sw))
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