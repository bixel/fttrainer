#! /usr/bin/env python

# since there's still py2 around
from __future__ import division, print_function

import argparse
import json

import numpy as np

import ROOT
# disable cmd line parsing before other ROOT deps are loaded
ROOT.PyConfig.IgnoreCommandLineOptions = True  # noqa
from root_pandas import read_root
from xgboost import XGBClassifier

from scripts.metrics import tagging_power_score, get_event_number


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, default=None)
    parser.add_argument('-i', '--input-file', type=str, default=None,
                        help="""Read in a preselected training tuple (created
                        e.g. by the crossval_training.py script) instead of
                        applying the selection to a full tuple. This option
                        will prevent average tagging power values to be
                        printed.""")
    parser.add_argument('--has-predictions', default=False, action='store_true',
                        help="""The input fill already has predictions. Only
                        calculate its tagging power.""")
    parser.add_argument('-o', '--output-file', type=str, default=None,
                        help="""Add predictions of the model to the tuple. Note
                        that this is NOT done in a cross-validated manner.""")
    parser.add_argument('-s', '--save-model', type=str, default=None,
                        help="""Write the XGBoost model to disk.""")
    parser.add_argument('--stop', type=int, default=None,
                        help="""The read_root stop argument.""")
    parser.add_argument('--verbose', default=False, action='store_true',
                        help="""Don't be silent while training.""")
    parser.add_argument('--additional-selection', default='', type=str,
                        help="""Define an additional selection string""")
    return parser.parse_args()


def parse_config(filename):
    if filename is None:
        return None
    with open(filename, 'r') as f:
        return json.load(f)


def print_tp(df, args, config):
    print('{:.2f}%'.format(100 * tagging_power_score(df, config, etas='probas')))


def add_predictions(args, config):
    print('Reading data...', end='', flush=True)
    df = read_root(args.input_file, stop=args.stop)
    df['target'] = df.eval(config['target_eval'])
    for col in df.columns:
        if df[col].dtype == np.uint64:
            df[col] = df[col].astype(np.int64)
    if args.additional_selection:
        df.query(args.additional_selection, inplace=True)
    print(' done.')

    xgb_kwargs = config.get('xgb_kwargs', {})
    model = XGBClassifier(**xgb_kwargs, nthread=config.get('n_jobs'),
                          silent=not args.verbose)

    mva_features = config['mva_features']
    X = df[mva_features]
    y = df.target
    print('Starting training...', end='', flush=True)
    model.fit(X, y)
    print(' done.')

    print('Predicting now...', end='', flush=True)
    df['probas'] = model.predict_proba(X)[:, 1]
    print(' done.')

    if args.save_model:
        print('Saving model to `{}`...'.format(
            args.save_model), end='', flush=True)
        model.booster().save_model(args.save_model)
        print(' done.')

    if args.output_file:
        print('Saving tuple to `{}`...'.format(
            args.output_file), end='', flush=True)
        df.to_root(args.output_file)
        print(' done.')

    return df


def main():
    args = parse_args()
    config = parse_config(args.config_file)

    if args.has_predictions:
        df = read_root(args.input_file)
    else:
        df = add_predictions(args, config)

    print_tp(df, args, config)


if __name__ == '__main__':
    main()
