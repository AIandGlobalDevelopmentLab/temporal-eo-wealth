import os
from glob import glob
import pandas as pd


def create_full_tfrecords_paths(data_path):
    """
    Creates sorted list with paths to all tf-record files in dhs_clusters.csv
    :param data_path: str, path to directory containing data files
    :return: A list with paths to tf-record files
    """
    tfrecord_paths = []
    df = pd.read_csv(data_path + '/dhs_clusters.csv', float_precision='high', index_col=False)
    surveys = list(df.groupby(['country', 'survey_start_year']).groups.keys())
    for (country, year) in surveys:
        glob_path = os.path.join(data_path + '/dhs_tfrecords/', country + '_'
                                 + str(year), '*.tfrecord.gz')
        survey_files = glob(glob_path)
        tfrecord_paths.extend(sorted(survey_files))
    return tfrecord_paths