import tensorflow as tf
import numpy as np
import argparse
import json
import os
import pickle
from pprint import pprint
from typing import Any

import data_handling

def create_hists(**params: Any) -> None:
    '''
    params is a dict with keys matching the arguments from _parse_args()
    '''

    # print all of the flags
    pprint(params)

    # set the random seeds
    seed = params['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    tf.keras.backend.set_epsilon(1e-4)

    data_dir = params['data_dir']
    eval_folds = params['eval_folds']
    model_fold = params['model_fold']
    fold_config = params['fold_config']
    save_dir = params['save_dir']
    save_dir = os.path.join(save_dir, fold_config, model_fold)

    # get all tfrecords
    tfrecord_files = np.asarray(data_handling.create_full_tfrecords_paths(data_dir))

    # get train, val, test fold
    folds_file_path = os.path.join(data_dir, 'folds', fold_config + '.pkl')
    with open(folds_file_path, 'rb') as pickle_file:
        cv_indices = pickle.load(pickle_file)

    # get band stats
    stats_file_path = os.path.join(data_dir, 'band_stats', fold_config + '.json')
    with open(stats_file_path) as band_stats_file:
        all_folds_band_stats = json.load(band_stats_file)
    band_stats = all_folds_band_stats[model_fold]

    # Get and save features for all folds
    for fold in eval_folds:
        fold_indices = cv_indices[fold]['test']
        fold_files = tfrecord_files[fold_indices]

        ds = data_handling.get_inference_dataset(fold_files, batch_size=0, 
                            band_stats=band_stats, labeled=True)
        
        hists, labels, sample_indices = get_hists(ds)
        
        save_fold_hists(save_dir, fold, hists, labels, sample_indices)


def get_hists(ds):
    img_hists = []
    labels = []
    sample_indices = []

    band_bin_centers = np.arange(0, 1, 0.01)
    band_bin_edges = np.concatenate([
        [-1e5],
        band_bin_centers,
        [1e5]
    ])

    for x, y in ds:
        i = x['frame_index']
        img = x['model_input'][i]
        img_hist, label = get_hist(img, y, band_bin_edges)
        img_hists.append(img_hist)
        labels.append(label)
        sample_indices.append(x['sample_index'])
        
    img_hists = np.stack(img_hists)
    labels = np.asarray(labels)
    sample_indices = np.asarray(sample_indices)
    
    return img_hists, labels, sample_indices


def get_hist(img, label, band_bin_edges):
    bands = tf.reshape(img, (224 * 224, 8)).numpy()
    img_hist = np.apply_along_axis(lambda band: np.histogram(band, bins=band_bin_edges)[0], 0, bands).T
    return img_hist, label.numpy()


def save_fold_hists(save_dir, fold, hists, labels, indices):
    # Create save_dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create dir for fold
    fold_save_dir = os.path.join(save_dir, fold)
    os.makedirs(fold_save_dir)

    # Save variables
    hists_path = os.path.join(fold_save_dir, 'hists.npy')
    with open(hists_path, 'wb') as f:
        np.save(f, hists)
    print(f'Saved histograms with to {hists_path}')

    labels_path = os.path.join(fold_save_dir, 'labels.npy')
    with open(labels_path, 'wb') as f:
        np.save(f, labels)
    print(f'Saved labels with shape {labels.shape} to {labels_path}')

    indices_path = os.path.join(fold_save_dir, 'indices.npy')
    with open(indices_path, 'wb') as f:
        np.save(f, indices)
    print(f'Saved indices with shape {indices.shape} to {indices_path}')


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Creates features from model for HP tuning')

    # paths
    parser.add_argument(
        '--data_dir', default='/mimer/NOBACKUP/groups/globalpoverty1/data', 
        required=False, help='path to data directory')
    parser.add_argument(
        '--save_dir', required=True,
        help='path to save directory')
    parser.add_argument(
        '--model_fold', default='A', choices=['A', 'B', 'C', 'D', 'E'],
        help='CV fold used for training model')
    parser.add_argument(
        '--eval_folds', default='ABCDE',
        help='CV folds to make predictions for')
    parser.add_argument(
        '--fold_config', default='incountry', choices=['incountry', 'ooc', 'oots'],
        help='What folds was used during training. Either \'incountry\', \'ooc\' or \'oots\'.')

    # Misc
    parser.add_argument(
        '--seed', type=int, default=123,
        help='seed for random initialization and shuffling')
    return parser.parse_args()

# Example:
# singularity run -B /mimer/NOBACKUP/groups/globalpoverty1/ /cephyr/NOBACKUP/groups/globalpoverty1/singularity_imgs/container_latest.sif create_hists.py --save_dir=/cephyr/NOBACKUP/groups/globalpoverty1/data/hists --model_fold=A --fold_config=incountry --eval_folds=AB

if __name__ == '__main__':
    args = _parse_args()
    create_hists(**vars(args))
    exit(0)


