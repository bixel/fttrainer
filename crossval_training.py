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

from scripts.data_preparation import NSplit
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
    parser.add_argument('--overwrite', default=False,
                        action='store_true', help="""Should the output file
                        be overwritten if it exists?""")
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
        tree = rootfile.Get(kwargs.get('key'))
        entries += tree.GetEntries()
    maxslices = args.max_slices
    chunksize = kwargs['chunksize']
    total = (maxslices
             if maxslices is not None and maxslices < (entries / chunksize)
             else (entries / chunksize))

    print(total * chunksize, 'events.')

    # clean output file if it exists
    if args.output_file and os.path.isfile(args.output_file):
        if args.overwrite:
            os.remove(args.output_file)
        else:
            print('file `{}` exists! set overwrite argument to overwrite '
                  'the file.'.format(args.output_file))
            sys.exit(1)

    merged_training_df = None

    # loop over tuple and fill training variables
    for df in tqdm(
            islice(read_root(files, flatten=True, **kwargs), maxslices),
            total=total):
        df['target'] = df.eval(config['target_eval'])

        # set a proper index
        df.set_index(['runNumber', 'eventNumber', '__array_index'],
                     inplace=True, drop=True)

        # apply selections
        selected_df = df
        selections = config['selections']
        maxQ = 10
        for s in [selections[i * maxQ:i * maxQ + maxQ]
                  for i in range(int(len(selections) / maxQ) + 1)]:
            if len(s):
                selected_df.query(' and '.join(s), inplace=True)

        # select n max pt particles
        sorting_feature = config['sorting_feature']
        nMax = config.get('particles_per_event', 1)
        grouped = selected_df.groupby(['runNumber', 'eventNumber'], sort=False)
        # calculate indices of the top n rows in each group;
        # depending on how many particles are found in each group, the index
        # needs to be reset. This seems to be a bug and might be fixed in 0.20
        # see pandas github issue #15297
        if grouped[sorting_feature].count().max() > nMax:
            indices = grouped[sorting_feature].nlargest(nMax).reset_index([0, 1]).index
        else:
            indices = grouped[sorting_feature].nlargest(nMax).index
        max_df = selected_df.loc[indices]

        if args.output_file:
            max_df.reset_index().to_root(args.output_file, mode='a')

        # append this chunk to the training dataframe
        merged_training_df = pd.concat([merged_training_df, max_df])

    return merged_training_df


def print_avg_tagging_info(df, config):
    df = df.groupby(['runNumber', 'eventNumber']).first()
    total_event_number = get_event_number(config)
    event_number = df.groupby(
        ['runNumber', 'eventNumber']).SigYield_sw.first().sum()
    event_number = ufloat(event_number, np.sqrt(event_number))
    efficiency = event_number / total_event_number
    wrong_tag_number = np.sum(df.SigYield_sw * ~df.target)
    wrong_tag_number = ufloat(wrong_tag_number, np.sqrt(wrong_tag_number))
    avg_omega = wrong_tag_number / event_number
    avg_omega = ufloat(avg_omega.n, avg_omega.n / np.sqrt(event_number.n))
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


def get_event_number(config):
    files = [config['filepath'] + f for f in config['files']]
    df = read_root(files, key=config['pandas_kwargs']['key'],
                   columns=['SigYield_sw', 'nCandidate'])
    return df[df.nCandidate == 0].SigYield_sw.sum()


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
        merged_training_df.set_index(['runNumber', 'eventNumber',
                                      '__array_index'], inplace=True)
    else:
        merged_training_df = read_full_files(args, config)

    print_avg_tagging_info(merged_training_df, config)

    mva_features = config['mva_features']
    efficiency = merged_training_df.SigYield_sw.sum() / get_event_number(config)

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

            max_pt_particles = df3.groupby(['runNumber', 'eventNumber']).head(1)

            score = tagging_power_score(max_pt_particles.calib_probas,
                                        efficiency=efficiency,
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
