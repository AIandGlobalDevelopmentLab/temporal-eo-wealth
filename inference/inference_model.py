import argparse
import json
import os
import pickle
from pprint import pprint
from typing import Any, Dict

import numpy as np
import tensorflow as tf

import pandas as pd

import data_handling
from models.wrap_model import get_wrapped_model


def infer_model(**params: Any) -> None:
    '''
    params is a dict with keys matching the arguments from _parse_args()
    '''

    # print all of the flags
    pprint(params)

    data_dir = params['data_dir']
    fold_config = params['fold_config']
    model_name = params['model_name']

    # set the random seeds
    seed = params['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    tf.keras.backend.set_epsilon(1e-4)

    # get all tfrecords
    tfrecord_files = np.asarray(data_handling.create_full_tfrecords_paths(data_dir))
    
    # get train, val, test fold
    folds_file_path = os.path.join(data_dir, 'folds', fold_config + '.pkl')
    with open(folds_file_path, 'rb') as pickle_file:
        content = pickle.load(pickle_file)

    # get band stats
    stats_file_path = os.path.join(data_dir, 'band_stats', fold_config + '.json')
    with open(stats_file_path) as band_stats_file:
        all_folds_band_stats = json.load(band_stats_file)
    
    fold_results = []
    folds = 'ABCDE'
    
    for fold in folds:
        print('Processing fold', fold)
        model_path = os.path.join(data_dir, 'features', fold_config, model_name, fold, 'tuned_model')
    
        test_indices = content[fold]['test']
        num_test = len(test_indices)
        print('num_test:', num_test)
        test_files = tfrecord_files[test_indices]
        
        band_stats = all_folds_band_stats[fold]
        ds = data_handling.get_inference_dataset(test_files, 
            batch_size=16, band_stats=band_stats, labeled=False)
        fold_result = infer_model_fold(model_path, ds)
        fold_result['fold'] = fold
        fold_results.append(fold_result)
    
    result_df = pd.concat(fold_results).sort_index()
    result_df.to_csv(os.path.join(data_dir, 'results', fold_config, model_name + '.csv'), index=True)
        
        
def infer_model_fold(model_path, ds):
    model = get_wrapped_model(model_path)
    index = np.array([], dtype='int')

    results = {f'y_{i}': np.array([], dtype='float32') for i in range(10)}
    results['y_i'] = np.array([], dtype='float32')

    for x in ds:
        y = model(x)
        i = tf.range(x['frame_index'].shape[0])
        y_i = tf.gather_nd(y, indices=tf.stack([i, x['frame_index']], axis=1))
        
        ix = x['sample_index'].numpy()
        y_i = y_i.numpy()
        y = y.numpy()
        
        index = np.append(index, ix)
        results['y_i'] = np.append(results['y_i'], y_i)

        for i in range(10):
            results[f'y_{i}'] = np.append(results[f'y_{i}'], y[:, i])
        
    return pd.DataFrame(results, index=index)


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run end-to-end population based training.')

    # paths
    parser.add_argument(
        '--data_dir', default='/cephyr/NOBACKUP/groups/globalpoverty1/data', 
        required=False, help='path to data directory')
    parser.add_argument(
        '--model_name', required=True, choices=['resnet', 'bidirectional_resnet_lstm', 'bidirectional_resnet_lstm_10'],
        help='name of model')
    parser.add_argument(
        '--fold_config', default='incountry', choices=['incountry', 'ooc', 'oots'],
        help='What folds was used during training. Either \'incountry\', \'ooc\' or \'oots\'.')

    # Misc
    parser.add_argument(
        '--seed', type=int, default=123,
        help='seed for random initialization and shuffling')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    infer_model(**vars(args))
    exit(0)

