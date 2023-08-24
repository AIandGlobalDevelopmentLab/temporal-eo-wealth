import argparse
import json
import os
import pickle
from pprint import pprint
import time
import math
from typing import Any, List

import numpy as np
import tensorflow as tf
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_handling

IMAGE_SIZE = 224
N_YEAR_COMPOSITES = 10
N_DMSP_COMPOSITES = 8
N_VIIRS_COMPOSITES = N_YEAR_COMPOSITES - N_DMSP_COMPOSITES


def get_tfrecord_stats(example_proto):
    '''
    Args
    - example_proto: a tf.train.Example
    Returns:
    - sum of all pixel values per band (divided by number of pixels).
    - sum of all squared pixel values per band (divided by number of pixels).
    '''
    n_band_pixels = IMAGE_SIZE * IMAGE_SIZE * N_YEAR_COMPOSITES
    n_dmsp_pixels = IMAGE_SIZE * IMAGE_SIZE * N_DMSP_COMPOSITES
    n_viirs_pixels = IMAGE_SIZE * IMAGE_SIZE * N_VIIRS_COMPOSITES
    bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']
    keys_to_features = {}
    for band in bands:
        keys_to_features[band] = tf.io.FixedLenFeature(shape=[n_band_pixels], dtype=tf.float32)
    ex = tf.io.parse_single_example(example_proto, features=keys_to_features)

    stats = {}
    for band in bands[:-1]:
        band_ex = tf.nn.relu(ex[band])
        band_ex_squared = tf.math.square(band_ex)
        stats[band + '_SUM'] = tf.math.divide(tf.math.reduce_sum(band_ex), n_band_pixels)
        stats[band + '_SQUARED_SUM'] = tf.math.divide(tf.math.reduce_sum(band_ex_squared), n_band_pixels)
        stats[band + '_MIN'] = tf.math.reduce_min(band_ex)
        stats[band + '_MAX'] = tf.math.reduce_max(band_ex)

    nl_band = 'NIGHTLIGHTS'
    nl_ex = tf.nn.relu(ex[nl_band])
    nl_ex = tf.reshape(nl_ex, (N_YEAR_COMPOSITES, IMAGE_SIZE, IMAGE_SIZE))
    nl_split = tf.split(nl_ex, [N_DMSP_COMPOSITES, N_VIIRS_COMPOSITES], axis=0)

    dmsp_ex = nl_split[0]
    dmsp_ex_squared = tf.math.square(dmsp_ex)
    stats['DMSP_SUM'] = tf.math.divide(tf.math.reduce_sum(dmsp_ex), n_dmsp_pixels)
    stats['DMSP_SQUARED_SUM'] = tf.math.divide(tf.math.reduce_sum(dmsp_ex_squared), n_dmsp_pixels)
    stats['DMSP_MIN'] = tf.math.reduce_min(dmsp_ex)
    stats['DMSP_MAX'] = tf.math.reduce_max(dmsp_ex)

    viirs_ex = nl_split[1]
    viirs_ex_squared = tf.math.square(viirs_ex)
    stats['VIIRS_SUM'] = tf.math.divide(tf.math.reduce_sum(viirs_ex), n_viirs_pixels)
    stats['VIIRS_SQUARED_SUM'] = tf.math.divide(tf.math.reduce_sum(viirs_ex_squared), n_viirs_pixels)
    stats['VIIRS_MIN'] = tf.math.reduce_min(viirs_ex)
    stats['VIIRS_MAX'] = tf.math.reduce_max(viirs_ex)

    return stats


def get_stat_ds(tfrecord_files):
    '''Gets the dataset preprocessed and split into batches and epochs.
    Returns
    - dataset, each sample of the form {"model_input": img, "outputs_mask": one_hot_year}, {"masked_outputs": one_hot_label}
    '''
    # convert to individual records
    dataset = tf.data.TFRecordDataset(
        filenames=tfrecord_files,
        compression_type='GZIP',
        buffer_size=1024 * 1024 * 128,  # 128 MB buffer size
        num_parallel_reads=4)  # num_threads)
    # prefetch 2 batches at a time to smooth out the time taken to
    # load input files as we go through shuffling and processing
    dataset = dataset.prefetch(buffer_size=32)
    dataset = dataset.map(get_tfrecord_stats, num_parallel_calls=4)
    return dataset


def get_variance(tot_sum, tot_squared_sum, n_samples):
    mean = tot_sum/n_samples
    return (tot_squared_sum/n_samples) - (mean * mean)


def run_eval(tfrecord_files, fold_indices):
    # set bands
    stat_bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'DMSP', 'VIIRS']

    # TODO: loop over all instead
    # select which fold to use
    folds = ['A', 'B', 'C', 'D', 'E']

    stats_data = {}
    for fold in folds:
        train_indices = fold_indices[fold]['train']

        num_train = len(train_indices)
        train_files = tfrecord_files[train_indices]

        stat_ds = get_stat_ds(train_files)

        tot_sums = {}
        tot_squared_sums = {}
        tot_mins = {}
        tot_maxs = {}
        for band in stat_bands:
            tot_sums[band] = 0
            tot_squared_sums[band] = 0
            tot_mins[band] = np.inf
            tot_maxs[band] = -np.inf

        for elem in stat_ds.as_numpy_iterator():
            for band in stat_bands:
                tot_sums[band] += elem[band + '_SUM']
                tot_squared_sums[band] += elem[band + '_SQUARED_SUM']
                if elem[band + '_MIN'] < tot_mins[band]:
                    tot_mins[band] = elem[band + '_MIN']
                if elem[band + '_MAX'] > tot_maxs[band]:
                    tot_maxs[band] = elem[band + '_MAX']

        fold_stats_data = {
            'means': {},
            'stds': {},
            'mins': {},
            'maxs': {},
            'norm_mins': {},
            'norm_maxs': {}
        }
        for band in stat_bands:
            band_mean = tot_sums[band] / num_train
            fold_stats_data['means'][band] = band_mean
            band_var = get_variance(tot_sums[band], tot_squared_sums[band], num_train)
            band_std = math.sqrt(band_var)
            fold_stats_data['stds'][band] = band_std
            band_min = tot_mins[band].item()
            fold_stats_data['mins'][band] = band_min
            band_max = tot_maxs[band].item()
            fold_stats_data['maxs'][band] = band_max
            fold_stats_data['norm_mins'][band] = (band_min - band_mean) / band_std
            fold_stats_data['norm_maxs'][band] = (band_max - band_mean) / band_std

        stats_data[fold] = fold_stats_data
    
    return stats_data


def run_evals(
        data_dir: str,
        out_dir: str) -> None:
    '''
    Args
    - data_dir: str
    - ls_bands: List[str], the names of the bands in the TFRecords
    - stat_bands: List[str], the name of the bands to store stats for
    - out_dir: str, path to output directory for saving stats, must already exist
    - n_year_composites: int
    - n_dmsp_composites: int
    '''
    # ====================
    #    ERROR CHECKING
    # ====================
    assert os.path.exists(out_dir)

    # ====================
    #       DATASET
    # ====================

    # get all tfrecords
    tfrecord_files = np.asarray(data_handling.create_full_tfrecords_paths(data_dir))

    # Evaluate each of the three split-configurations
    for config in ['incountry', 'ooc', 'oots']:

        # get train, val, test fold
        fold_file_path = os.path.join(data_dir, 'folds', config + '.pkl')
        with open(fold_file_path, 'rb') as pickle_file:
            fold_indices = pickle.load(pickle_file)

        stats_data = run_eval(tfrecord_files, fold_indices)
        print(f'Band stats for {config}:')
        print(stats_data)

        with open(os.path.join(out_dir, config + '.json'), 'w') as outfile:
            json.dump(stats_data, outfile, indent=4)


def run_eval_wrapper(**params: Any) -> None:
    '''
    params is a dict with keys matching the arguments from _parse_args()
    '''
    start = time.time()
    print('Current time:', start)

    # print all of the flags
    pprint(params)

    # set the random seeds
    seed = params['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # create the output directory if needed
    out_dir = os.path.join(params['data_dir'], 'band_stats')
    print(f'Outputs directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    run_evals(
        data_dir=params['data_dir'],
        out_dir=out_dir
    )

    end = time.time()
    print('End time:', end)
    print('Time elapsed (sec.):', end - start)


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run end-to-end training.')

    # paths
    parser.add_argument(
        '--data_dir', default='/mimer/NOBACKUP/groups/globalpoverty1/data',
        help='path to data directory')

    # Misc
    parser.add_argument(
        '--seed', type=int, default=123,
        help='seed for random initialization and shuffling')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_eval_wrapper(**vars(args))
