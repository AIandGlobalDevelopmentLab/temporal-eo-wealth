import numpy as np
import pandas as pd
import argparse
import os
import scipy.spatial
import time
from pprint import pprint
from typing import Any
from baseline.knn import train_knn_logo

def create_hists(**params: Any) -> None:
    '''
    params is a dict with keys matching the arguments from _parse_args()
    '''

    # print all of the flags
    pprint(params)

    # set the random seeds
    seed = params['seed']
    np.random.seed(seed)

    data_dir = params['data_dir']
    model_fold = params['model_fold']
    fold_config = params['fold_config']
    save_plots = params['save_plots']
    
    # Load histogram data
    df = pd.read_csv(os.path.join(data_dir, 'dhs_clusters.csv'), float_precision='high', index_col=False)
    features, labels, indices, group_labels, test_groups = get_data(data_dir, fold_config, model_fold, df)
    group_names = np.unique(group_labels)

    N = len(labels)
    assert len(features) == N
    assert len(group_labels) == N

    # Precalculate distance matrix
    dists = get_dist_matrix(features)

    # Create dir for plots, if saveing
    if save_plots:
        plot_dir = os.path.join(data_dir, 'knn_plots', fold_config)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    start = time.time()
    test_preds = np.ones(len(df), dtype=np.float32)

    # Make separate model for each group in test fold
    for i, f in enumerate(test_groups):
        print('Group:', f)
        group_start = time.time()

        # Get test folds
        test_mask = (group_labels == f)
        test_indices = indices[test_mask]
        
        plot = os.path.join(plot_dir, f'{f}.png') if save_plots else None

        # Train KNN model with HP tuning and make precitions
        test_preds[test_indices] = train_knn_logo(
            dists=dists,
            features=features,
            labels=labels,
            group_labels=group_labels,
            cv_groups=[x for x in group_names if x != f],
            test_groups=[f],
            weights=None,
            plot=plot,
            group_names=group_names)
        
        elapsed = time.time() - group_start
        print(f' took {elapsed:.2f} seconds.')
    elapsed = time.time() - start
    print(f' took {elapsed:.2f} seconds.')

    print('Saving results...')
    test_indices = indices[np.argwhere(np.isin(group_labels, test_groups)).ravel()]
    result_df = pd.DataFrame({'y_i': test_preds[test_indices]}, index=test_indices)
    result_df['fold'] = model_fold
    result_df.to_csv(os.path.join(data_dir, 'hists', fold_config, model_fold, 'knn_preds.csv'), index=True)

def get_dist_matrix(features, distance_metric='cityblock'):
    print('Pre-computing distance matrix...', end='')
    start = time.time()
    dists = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(features, metric=distance_metric)
    )
    elapsed = time.time() - start
    print(f' took {elapsed:.2f} seconds.')
    return dists

def get_data(data_dir, fold_config, model_fold, df):

    # Get path
    hist_dir = os.path.join(data_dir, 'hists', fold_config, model_fold, '{}')

    X_list = []
    y_list = []
    ix_list = []
    group_list = []

    for i, fold in enumerate('ABCDE'):
        fold_dir = hist_dir.format(fold)
        fold_X = np.load(os.path.join(fold_dir, 'hists.npy'))
        X_list.append(fold_X)
        fold_y = np.load(os.path.join(fold_dir, 'labels.npy')).flatten()
        y_list.append(fold_y)
        fold_ix = np.load(os.path.join(fold_dir, 'indices.npy'))
        ix_list.append(fold_ix)
        if fold_config == 'ooc':
            fold_countries = df.loc[fold_ix, 'country'].values
            group_list.append(fold_countries)
            if fold == model_fold:
                test_groups = np.unique(fold_countries)
        else:
            group_list.append([i] * len(fold_y))
            if fold == model_fold:
                test_groups = [i]

    img_hists = np.concatenate(X_list)
    features = img_hists.reshape(len(img_hists), -1)
    labels = np.concatenate(y_list)
    indices = np.concatenate(ix_list)
    group_labels = np.concatenate(group_list)

    return features, labels, indices, group_labels, test_groups

def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Creates features from model for HP tuning')

    # paths
    parser.add_argument(
        '--data_dir', default='/cephyr/NOBACKUP/groups/globalpoverty1/data', 
        required=False, help='path to data directory')
    parser.add_argument(
        '--model_fold', default='A', choices=['A', 'B', 'C', 'D', 'E'],
        help='CV fold used for training model')
    parser.add_argument(
        '--fold_config', default='incountry', choices=['incountry', 'ooc', 'oots'],
        help='What folds was used during training. Either \'incountry\', \'ooc\' or \'oots\'.')
    parser.add_argument(
        '--save_plots', action='store_true',
        help='Save plots of the HP tuning for KNN')

    # Misc
    parser.add_argument(
        '--seed', type=int, default=123,
        help='seed for random initialization and shuffling')
    return parser.parse_args()

# Example:
# singularity run -B /mimer/NOBACKUP/groups/globalpoverty1/ /cephyr/NOBACKUP/groups/globalpoverty1/singularity_imgs/container_latest.sif knn_train.py --model_fold=A --fold_config=ooc --save_plots

if __name__ == '__main__':
    args = _parse_args()
    create_hists(**vars(args))
    exit(0)


