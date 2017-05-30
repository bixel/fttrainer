#! /usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve

import ROOT
# disable cmd line parsing before other ROOT deps are loaded
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, default=None)
    parser.add_argument('-m', '--max-slices', type=int, default=None)
    parser.add_argument('-p', '--plot', type=str, default=None,
                        help='Create a plot of the average ROC curve and the 1'
                             'sigma area around the curve.')
    parser.add_argument('-i', '--input-file', type=str, default=None,
                        help='Read in a preselected training tuple (created'
                        ' e.g. by this script) instead of applying the'
                        ' selection to a full tuple. This option will prevent'
                        ' average tagging power values to be printed.')
    parser.add_argument('-o', '--output-file', type=str, default=None,
                        help='Write a .root file containing only the training '
                        'ranches, as well as run number information. '
                        'This can be used for a final BDT training '
                        '+ calibration.')
    return parser.parse_args()


def parse_config(filename):
    if filename is None:
        return None
    with open(filename, 'r') as f:
        return json.load(f)


def read_full_files(args, config):
    # read dataset, create a list of absolute filenames
    files = [os.path.join(config.get('filepath', ''), f)
             for f in config['files']]
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

    merged_training_df = None

    # loop over tuple and fill training variables
    for df in tqdm(
            islice(read_root(files, flatten=True, **kwargs), maxslices),
            total=total):
        taggingParticleBranchPrefix = config['tagging_particle_prefix']

        # rename the tagging particle branches
        df.rename(columns=dict(zip(df.columns,
                                   [c.replace(taggingParticleBranchPrefix, 'tp').replace('-', '_')
                                    for c in df.columns])),
                  inplace=True)
        if 'invert_target' in config and config['invert_target']:
            df['target'] = np.sign(df.B_ID) != np.sign(df.tp_ID)
        else:
            df['target'] = np.sign(df.B_ID) == np.sign(df.tp_ID)

        # read features and selections
        selection_query = ' and '.join(['tp_' + f for f in config['selections']])

        # apply selections
        selected_df = df.query(selection_query)

        # select max pt particles
        sorting_feature = ('tp_' + config['sorting_feature'])
        max_df = selected_df.iloc[selected_df
                                  .groupby(['runNumber', 'eventNumber'])[sorting_feature]
                                  .idxmax()].copy()
        max_df['probas'] = 0.5
        max_df['calib_probas'] = 0.5

        # append this chunk to the training dataframe
        merged_training_df = pd.concat([merged_training_df, max_df])

    print_avg_tagging_info(merged_training_df, total * chunksize)

    return merged_training_df


def print_avg_tagging_info(selected_df, total_event_number):
    total_event_number = ufloat(total_event_number, np.sqrt(total_event_number))
    selected_event_number = get_event_number(selected_df)
    selected_event_number = ufloat(selected_event_number, np.sqrt(selected_event_number))
    efficiency = selected_event_number / total_event_number
    wrong_tag_number = np.sum(selected_df.SigYield_sw * ~selected_df.target)
    wrong_tag_number = ufloat(wrong_tag_number, np.sqrt(wrong_tag_number))
    avg_omega = wrong_tag_number / selected_event_number
    avg_omega = ufloat(avg_omega.n, avg_omega.n / np.sqrt(selected_event_number.n))
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


def main():
    args = parse_args()
    config = parse_config(args.config_file)
    if config is None:
        print('No configuration file is defined. '
              'Define one with `--config-file`.')
        sys.exit(1)

    # this will be the training dataframe
    if args.input_file:
        merged_training_df = read_root(args.input_file)
    else:
        merged_training_df = read_full_files(args, config)

        if args.output_file:
            merged_training_df.to_root(args.output_file)
            print('File {} has been written. Exiting now.'
                  .format(args.output_file))
            sys.exit()

    mva_features = ['tp_' + f for f in config['mva_features']]

    # build BDT model and train the classifier n_cv x 3 times
    xgb_kwargs = config['xgb_kwargs']
    n_jobs = config['n_jobs']

    bootstrap_scores = []
    bootstrap_d2s = []
    bootstrap_roc_curves = []
    nfold = config['n_cv']
    print('Starting bootstrapping.')
    pbar = tqdm(total=nfold * 6)
    for _ in range(nfold):
        # yield 3-fold split for CV
        df_sets = [merged_training_df.iloc[indices]
                   for indices in NSplit(merged_training_df)]
        # try to compensate for slow subset creation
        pbar.update(3)

        for i in range(3):
            df1, df2, df3 = (df_sets[i % 3].copy(),
                             df_sets[(i + 1) % 3].copy(),
                             df_sets[(i + 2) % 3].copy())
            model = XGBClassifier(nthread=n_jobs, **xgb_kwargs)
            model.fit(df1[mva_features], df1.target,
                      sample_weight=df1.SigYield_sw)

            probas = model.predict_proba(df2[mva_features])[:, 1]
            df2['probas'] = probas
            merged_training_df.loc[df2.index, 'probas'] = probas

            # calibrate
            calibrator = PolynomialLogisticRegression(power=3,
                                                      solver='lbfgs',
                                                      n_jobs=n_jobs)
            calibrator.fit(df2.probas.values.reshape(-1, 1), df2.target)

            probas = model.predict_proba(df3[mva_features])[:, 1]
            calib_probas = calibrator.predict_proba(probas)[:, 1]
            df3['calib_probas'] = calib_probas
            merged_training_df.loc[df3.index, 'calib_probas'] = calib_probas

            score = tagging_power_score(calib_probas,
                                        efficiency=0.07,
                                        sample_weight=df3.SigYield_sw)
            if args.plot is not None:
                fpr, tpr = roc_curve(df3.target, probas,
                                     sample_weight=df3.SigYield_sw)[:2]
                bootstrap_roc_curves.append([fpr, tpr])

            bootstrap_scores.append(score)
            bootstrap_d2s.append(d2_score(calib_probas,
                                          sample_weight=df3.SigYield_sw))
            pbar.update(1)
    pbar.close()

    # plot roc curve on request
    if args.plot is not None:
        print('Plotting ROC curves...', end=' ')
        curve_points = np.array(bootstrap_roc_curves)

        # hacky test for correct roc curve shapes
        min_roc_shape = np.min([len(a[0]) for a in curve_points])
        fprs, tprs = [], []
        for fpr, tpr in curve_points:
            fprs.append(fpr[:min_roc_shape])
            tprs.append(tpr[:min_roc_shape])
        fprs = np.array(fprs)
        tprs = np.array(tprs)
        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = (6, 6)
        plt.rcParams['font.size'] = 12
        plt.plot([0, 1], '--', label='random')
        plt.plot(fprs.mean(axis=0), tprs.mean(axis=0), label='Mean ROC curve')
        plt.fill_between(fprs.mean(axis=0),
                         tprs.mean(axis=0) - tprs.std(axis=0),
                         tprs.mean(axis=0) + tprs.std(axis=0),
                         label=r'$\pm 1 \sigma$ area',
                         alpha=0.4)
        plt.xlim(-0.05, 1.05)
        plt.ylim(0, 1.05)
        plt.text(1, 0.05, 'LHCb unofficial',
                 verticalalignment='bottom', horizontalalignment='right')
        plt.legend(loc='best')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        filename = (args.plot
                    if args.plot.endswith('.pdf')
                    else args.plot + '.pdf')
        plt.savefig(filename, bbox_inches='tight')
        print('done.')

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
