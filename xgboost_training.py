#! /usr/bin/env python

# since there's still py2 around
from __future__ import division, print_function

import argparse
import json

import ROOT
# disable cmd line parsing before other ROOT deps are loaded
ROOT.PyConfig.IgnoreCommandLineOptions = True  # noqa
from root_pandas import read_root
from xgboost import XGBClassifier

from scripts.metrics import tagging_power_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, default=None)
    parser.add_argument('-i', '--input-file', type=str, default=None,
                        help="""Read in a preselected training tuple (created
                        e.g. by the crossval_training.py script) instead of
                        applying the selection to a full tuple. This option
                        will prevent average tagging power values to be
                        printed.""")
    parser.add_argument('-o', '--output-file', type=str, default=None,
                        help="""Add predictions of the model to the tuple. Note
                        that this is NOT done in a cross-validated manner.""")
    parser.add_argument('-s', '--save-model', type=str, default=None,
                        help="""Write the XGBoost model to disk.""")
    return parser.parse_args()


def parse_config(filename):
    if filename is None:
        return None
    with open(filename, 'r') as f:
        return json.load(f)


def get_event_number(config):
    files = [config['filepath'] + f for f in config['files']]
    df = read_root(files, key=config['pandas_kwargs']['key'],
                   columns=['SigYield_sw', 'nCandidate'])
    return df[df.nCandidate == 0].SigYield_sw.sum()


def main():
    args = parse_args()
    config = parse_config(args.config_file)

    df = read_root(args.input_file)
    xgb_kwargs = config.get('xgb_kwargs', {})
    model = XGBClassifier(**xgb_kwargs)

    mva_features = config['mva_features']
    X = df[mva_features]
    y = df.target
    model.fit(X, y)
    df['probas'] = model.predict_proba(X)[:, 1]

    # df is expected to be sorted event-wise
    maxPtParticles = df.groupby(['runNumber', 'eventNumber']).head(1)
    print('{:.2f}%'.format(
        100 * tagging_power_score(
            maxPtParticles.probas, tot_event_number=get_event_number(config),
            sample_weight=maxPtParticles.SigYield_sw)))

    if args.save_model:
        model.booster().save_model(args.save_model)

    if args.output_file:
        df.to_root(args.output_file)

if __name__ == '__main__':
    main()
