from root_pandas import read_root
import pandas as pd

import numpy as np

from tqdm import tqdm

from itertools import chain


def read_and_select(dataframe_generator,
                    query,
                    chunksize=None,
                    max_rows=-1,
                    n_entries=None):
    total_length = 0
    read_chunks = []

    if max_rows > 0:
        total = max_rows
    elif n_entries:
        total = n_entries
    else:
        total = 1

    with tqdm(total=total) as pbar:
        for chunk in dataframe_generator:
            chunk.query(query, inplace=True)

            # append chunk to dataset
            read_chunks.append(chunk)
            total_length += len(chunk)

            pbar.update(len(chunk))

            if max_rows > 0 and total_length > max_rows:
                print('Max rows reached.')
                return pd.concat(read_chunks)

    return pd.concat(read_chunks)


def grouped_aggregate(dataframe, column='B_P', function=np.argmax):
    # Create a groupby object (B_P should be unique) and only select max
    # of given branch_name
    group = dataframe.groupby(column)
    dataframe = dataframe.loc[group[branch_name].agg(function).astype('int')]
    return datafram
