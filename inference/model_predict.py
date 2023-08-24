
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from pprint import pprint
from typing import Any
import data_handling

ROOT_DIR = os.path.dirname(__file__)  # folder containing this file


def run_evaluation(fold_indices: slice,
                   tfrecord_files: np.ndarray,
                   test_fold: str,
                   model_dir: str,
                   batch_size: int,
                   one_year_model: bool,
                   n_year_composites: int,
                   normalize: bool):
    test_files = tfrecord_files[fold_indices]
    fold_model = tf.keras.models.load_model(f'{model_dir}/{test_fold}')
    test_ds = data_handling.get_train_dataset(test_files, batch_size, labeled=False, one_year_model=one_year_model,
                                  n_year_composites=n_year_composites, normalize=normalize,
                                  max_epochs=1)
    fold_preds = fold_model.predict(test_ds)
    print('fold_preds.shape:', fold_preds.shape)
    print('len(fold_indices):', len(fold_indices))
    print('len(test_files):', len(test_files))
    return fold_preds.squeeze()


def run_evaluation_wrapper(**params: Any) -> None:
    '''
    params is a dict with keys matching the arguments from _parse_args()
    '''
    start = time.time()
    print('Current time:', start)

    # print all of the flags
    pprint(params)

    # parameters that might be 'None'
    none_params = ['ls_bands']
    for p in none_params:
        if params[p] == 'None':
            params[p] = None

    # set the random seeds
    seed = params['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Get test folds
    # TODO: Fix incountry folds file
    with open(params['data_dir'] + '/new_dhs_incountry_folds.pkl', 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    test_folds = {}
    for fold in content:
        test_folds[fold] = content[fold]['test']

    # Init output df from dhs_clusters.csv
    df = pd.read_csv(params['data_dir'] + '/dhs_clusters.csv')
    df = df.sort_values(['country', 'year']).reset_index(drop=True)  # Sort by country, year since csv is unsorted
    for fold in test_folds:
        df.loc[df.index.isin(test_folds[fold]), 'fold_model'] = fold

    # create the output directory if needed
    # TODO: Create more descriptive name
    full_experiment_name = params['experiment_name']

    # get all tfrecords
    tfrecord_files = np.asarray(data_handling.create_full_tfrecords_paths(params['data_dir']))

    column_names = []
    for i in range(params['n_year_composites']):
        column_names.append(f'pred_{i}')

    df_filepath = os.path.join(params['model_dir'], 'dhs_cluster_predictions.csv')
    fold_dfs = []
    for fold in ['B', 'C', 'D', 'E']:
        print(f'Evaluating fold {fold}...')
        fold_indices = test_folds[fold]
        fold_df = df.iloc[fold_indices, :]
        fold_preds = run_evaluation(fold_indices=fold_indices, tfrecord_files=tfrecord_files, test_fold=fold,
                                    model_dir=params['model_dir'], batch_size=params['batch_size'],
                                    one_year_model=False,  # TODO: params['one_year_model'],
                                    n_year_composites=params['n_year_composites'], normalize=params['normalize'])
        fold_pred_df = pd.DataFrame(fold_preds, columns=column_names)
        fold_df = pd.concat([fold_df, fold_pred_df.set_index(fold_indices)], axis=1, join='inner')
        fold_df.to_csv(os.path.join(params['model_dir'], f'{fold}_dhs_cluster_predictions.csv'), index=False)
        fold_dfs.append(fold_df)

    pred_df = pd.concat(fold_dfs, axis=0)
    pred_df = pred_df.sort_index()

    df_filepath = os.path.join(params['model_dir'], 'dhs_cluster_predictions.csv')
    pred_df.to_csv(df_filepath, index=False)

    end = time.time()
    print('End time:', end)
    print('Time elapsed (sec.):', end - start)


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Evaluates model on test data.')

    # paths
    parser.add_argument(
        '--experiment_name', default='new_experiment',
        help='name of experiment being run')
    parser.add_argument(
        '--data_dir', default=os.path.join(ROOT_DIR, 'data/'),
        help='path to data directory')
    parser.add_argument(
        '--model_dir', default=os.path.join(ROOT_DIR, 'data/'),
        help='path to directory with models')

    # learning parameters
    parser.add_argument(
        '--label_name', default='iwi',
        help='name of label to use from the TFRecord files')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='batch size')

    # data params
    parser.add_argument(
        '--ls_bands', choices=[None, 'rgb', 'ms'],
        help='Landsat bands to use')
    parser.add_argument(
        '--n_year_composites', type=int, default=10,
        help='The number of year composites to use')
    parser.add_argument(
        '--normalize', action='store_true',
        help='Normalize the satellite data')

    # Misc
    parser.add_argument(
        '--seed', type=int, default=123,
        help='seed for random initialization and shuffling')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_evaluation_wrapper(**vars(args))
