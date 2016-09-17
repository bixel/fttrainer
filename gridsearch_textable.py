#! /usr/bin/env python

import json
import argparse

FEATURE_ORDER = [f + '_cut' for f in [
    'PT',
    'P',
    'phiDistance',
    'TRGHP',
    'IPPUs',
    'TRCHI2DOF',
    'PROBNNmu',
]]
NUM_COLS = 5


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='F', type=str, nargs='+',
                        help='The file with grid_search results')
    return parser.parse_args()


if __name__ == '__main__':
    args = setup_args()
    if args.file is None or len(args.file) > 1:
        raise("Only one file can be processed, pleaaase.")
    else:
        filename = args.file[0]

    results = {}
    with open(filename) as f:
        results = json.load(f)

    rows = {var: [] for var in FEATURE_ORDER}
    rows['tagging_power'] = []
    for score in results['scores'][:NUM_COLS]:
        for feature in FEATURE_ORDER:
            rows[feature] += [score[2][feature]]
        rows['tagging_power'] += [100 * score[0]]

    for feature in FEATURE_ORDER + ['tagging_power']:
        print('{:<16}'.format(feature), end=' & ')
        print(*rows[feature], sep=' & ', end=' ')
        print(r'\\')
