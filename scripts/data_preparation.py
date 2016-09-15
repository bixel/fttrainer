from root_pandas import read_root
import pandas as pd

import numpy as np

from tqdm import tqdm

from itertools import chain


def get_single_tagging_particles(dataframe_generator,
                                 target_function,
                                 chunksize,
                                 n_entries=None,
                                 branch_name='B_OS_Muon_PT'):
    full_data = pd.DataFrame()

    if n_entries:
        total = n_entries/chunksize
    else:
        total = None

    for chunk in tqdm(dataframe_generator,
                      total=total):
        # Define data target with target_function
        chunk['target'] = target_function(chunk)

        # Create a groupby object (B_P should be unique) and only select max
        # of given branch_name
        group = chunk.groupby('B_P')
        chunk = chunk.loc[group[branch_name].agg(np.argmax).astype('int')]

        # append chunk to dataset
        full_data = full_data.append(chunk.copy(deep=True))

    return full_data


def concat_df_chunks(filenames, chunksize, **kwargs):
    return chain(
        *(read_root(f, chunksize=chunksize, **kwargs) for f in filenames)
    )

class NSplit(object):
    def __init__(self, df, splits=3, shuffle=True):
        self.df = df
        self.df.reset_index(inplace=True, drop=True)
        self.unique_events = self.df.event_id.unique()
        self.raw_indices = np.arange(len(df))
        if shuffle:
            np.random.shuffle(self.unique_events)
        self.raw_index_sets = np.array_split(self.unique_events, splits)

    def __iter__(self):
        for index_set in self.raw_index_sets:
            yield self.raw_indices[self.df.event_id.isin(index_set).values]
