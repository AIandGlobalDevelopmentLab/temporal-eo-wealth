import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import TimeDistributed, Input

import argparse
import json
import os
import pickle
from pprint import pprint
from typing import Any

import numpy as np
import pandas as pd

import data_handling
from models.wrap_model import MakeWindowsLayer

N_FRAMES_IN_TS_IMG = 10

def infer_model(**params: Any) -> None:
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

    base_model_path = params['base_model_path']
    data_dir = params['data_dir']
    model_fold = params['model_fold']
    fold_config = params['fold_config']
    eval_folds = params['eval_folds']
    save_dir = params['save_dir']
    save_csv = params['save_csv']
    single_frame_model = params['single_frame_model']
    ten_frame_model = params['ten_frame_model']

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
    
    # Get feature model
    if single_frame_model:
        feat_model = create_single_frame_feat_model(base_model_path)
    elif ten_frame_model:
        feat_model = create_ten_frame_feat_model(base_model_path)
    else:
        feat_model = create_window_feat_model(base_model_path)

    # Get and save features for all folds
    for fold in eval_folds:
        print(f'Processing fold {fold}...')
        fold_indices = cv_indices[fold]['test']
        fold_files = tfrecord_files[fold_indices]
        fold_ds = data_handling.get_inference_dataset(fold_files, batch_size=16, 
                    band_stats=band_stats, labeled=True)
        if single_frame_model:
            fold_ds = mask_single_frame_ds(fold_ds)
            
        features, labels, weights, sample_indices = get_fold_features(feat_model, fold_ds, single_frame_model, ten_frame_model)
        save_fold_features(save_dir, fold, features, labels, weights, sample_indices)
        
    if save_csv:
        save_feature_csv(save_dir, eval_folds)

def create_single_frame_feat_model(path):
    # load base model
    base_model = load_model(path)
    feat_model = Model(inputs=base_model.input, outputs=base_model.get_layer('combined_resnet_output').output)
    return feat_model

def create_ten_frame_feat_model(path):
    #load base model
    base_model = load_model(path)
    feat_model = Model(inputs=base_model.input, outputs=base_model.get_layer('frame_features').output)
    return feat_model

def create_window_feat_model(path):
    # load base model
    base_model = load_model(path)
    base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('frame_features').output)

    # Repackage as feat_model
    window_size = base_model.input_shape[1]
    input_shape = tuple([N_FRAMES_IN_TS_IMG]) + base_model.input_shape[2:]
    inputs = Input(shape=input_shape, name='model_input')
    x = MakeWindowsLayer(window_size)(inputs)
    x = TimeDistributed(base_model, name='{}_frame_model'.format(window_size))(x)
    feat_model = Model(inputs=inputs, outputs=x, name='wrapped_{}_frame_feat_model'.format(window_size))
    return feat_model

def get_fold_features(feat_model, fold_ds, single_frame_model, ten_frame_model):
    features = []
    weights = []
    labels = []
    sample_indices = []
    window_size = feat_model.layers[-1].output.shape[-2]

    # Iterate data
    for x, y in fold_ds:
        # Make batch prediction
        feat = feat_model(x).numpy()
        batch_size = feat.shape[0]
        # Iterate over batch samples
        for i in range(batch_size):
            if single_frame_model:
                get_single_frame_features(feat, x, y, i, features, labels, sample_indices, weights)
            elif ten_frame_model:
                get_ten_frame_features(feat, x, y, i, features, labels, sample_indices, weights)
            else:
                get_window_features(feat, x, y, i, features, labels, sample_indices, weights, window_size)

    # Turn into np arrays
    features = np.stack(features)
    labels = np.asarray(labels)
    weights = np.asarray(weights)
    sample_indices = np.asarray(sample_indices)

    return features, labels, weights, sample_indices

def mask_single_frame_ds(ds):
    ds = ds.unbatch()
    ds = ds.map(mask_data)
    return ds.batch(16)
    
def mask_data(x, y):
    new_x = {
        'all_bands_inputs': x['model_input'][x['frame_index']],
        'sample_index': x['sample_index']
    }
    return new_x, y

def get_window_features(feat, x, y, i, feat_list, label_list, ind_list, w_list, window_size):
    # Get sample data
    feat_i = feat[i]
    y_i = y[i, 0].numpy()
    frame_index_i = x['frame_index'][i].numpy()
    sample_index_i = x['sample_index'][i].numpy()

    # Create sub sample indices
    start = max(0, frame_index_i - (window_size - 1))
    end = min(frame_index_i + 1, N_FRAMES_IN_TS_IMG - window_size + 1)
    out_indices = range(start, end)
    n_sub_samples = len(out_indices)

    # Add sub samples
    for j in out_indices:
        feat_list.append(feat_i[j, frame_index_i - j])
        label_list.append(y_i)
        ind_list.append(sample_index_i)
        w_list.append(1 / n_sub_samples)

def get_single_frame_features(feat, x, y, i, feat_list, label_list, ind_list, w_list):
    # Get sample data
    feat_i = feat[i]
    y_i = y[i, 0].numpy()
    sample_index_i = x['sample_index'][i].numpy()
    
    feat_list.append(feat_i)
    label_list.append(y_i)
    ind_list.append(sample_index_i)
    w_list.append(1.0)

def get_ten_frame_features(feat, x, y, i, feat_list, label_list, ind_list, w_list):
    # Get sample data
    frame_index_i = x['frame_index'][i].numpy()
    sample_index_i = x['sample_index'][i].numpy()
    feat_i = feat[i][frame_index_i]
    y_i = y[i, 0].numpy()
    
    feat_list.append(feat_i)
    label_list.append(y_i)
    ind_list.append(sample_index_i)
    w_list.append(1.0)

def save_fold_features(save_dir, fold, features, labels, weights, indices):
    # Create save_dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create dir for fold
    fold_save_dir = os.path.join(save_dir, fold)
    os.makedirs(fold_save_dir)

    # Save variables
    features_path = os.path.join(fold_save_dir, 'features.npy')
    with open(features_path, 'wb') as f:
        np.save(f, features)
    print(f'Saved features with shape {features.shape} to {features_path}')

    labels_path = os.path.join(fold_save_dir, 'labels.npy')
    with open(labels_path, 'wb') as f:
        np.save(f, labels)
    print(f'Saved labels with shape {labels.shape} to {labels_path}')

    indices_path = os.path.join(fold_save_dir, 'indices.npy')
    with open(indices_path, 'wb') as f:
        np.save(f, indices)
    print(f'Saved indices with shape {indices.shape} to {indices_path}')

    weights_path = os.path.join(fold_save_dir, 'weights.npy')
    with open(weights_path, 'wb') as f:
        np.save(f, weights)
    print(f'Saved weights with shape {weights.shape} to {weights_path}')
    
def save_feature_csv(save_dir, eval_folds):
    X = []
    y = []
    w = []
    ixs = []

    for fold in eval_folds:
        X.append(np.load(os.path.join(save_dir, fold, 'features.npy')))
        y.append(np.load(os.path.join(save_dir, fold, 'labels.npy')))
        w.append(np.load(os.path.join(save_dir, fold, 'weights.npy')))
        ixs.append(np.load(os.path.join(save_dir, fold, 'indices.npy')))

    # Get save folds
    save_folds = [[fold] * len(y) for fold, y in zip(eval_folds, y)]
    save_folds = sum(save_folds, [])

    X = np.concatenate(X)
    y = np.concatenate(y)
    w = np.concatenate(w)
    ixs = np.concatenate(ixs)

    df = pd.DataFrame(X, index=ixs)
    df['y'] = y
    df['w'] = w
    df['fold'] = save_folds
    df = df.sort_index()
    df.to_csv(os.path.join(save_dir, 'features.csv'))

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
        '--base_model_path', required=True,
        help='path to model for inference')
    parser.add_argument(
        '--save_dir', required=True,
        help='path to save directory')
    parser.add_argument(
        '--model_fold', default='A', choices=['A', 'B', 'C', 'D', 'E'],
        help='CV fold used for traing model')
    parser.add_argument(
        '--eval_folds', default='ABCDE',
        help='CV folds to make predictions for')
    parser.add_argument(
        '--fold_config', default='incountry', choices=['incountry', 'ooc', 'oots'],
        help='What folds was used during training. Either \'incountry\', \'ooc\' or \'oots\'.')
    parser.add_argument(
        '--save_csv', action='store_true',
        help='Save the results as a csv')
    parser.add_argument(
        '--single_frame_model', action='store_true',
        help='Is the model a single frame model?')
    parser.add_argument(
        '--ten_frame_model', action='store_true',
        help='Is the model a ten frame model?')

    # Misc
    parser.add_argument(
        '--seed', type=int, default=123,
        help='seed for random initialization and shuffling')
    return parser.parse_args()

# Example:
# singularity run -B /mimer/NOBACKUP/groups/globalpoverty1/ \
# /cephyr/NOBACKUP/groups/globalpoverty1/singularity_imgs/container_latest.sif \
# to_features.py --base_model_path=/cephyr/users/markpett/Alvis/satellite_poverty_prediction/best_model_test \
# --save_dir=/cephyr/NOBACKUP/groups/globalpoverty1/data/best_model_test_features \
# --model_fold=A --fold_config=incountry

# singularity run -B /mimer/NOBACKUP/groups/globalpoverty1/ /cephyr/NOBACKUP/groups/globalpoverty1/singularity_imgs/container_latest.sif to_features.py --base_model_path=/cephyr/users/markpett/Alvis/satellite_poverty_prediction/best_model_test --save_dir=/cephyr/NOBACKUP/groups/globalpoverty1/data/best_model_test_features --model_fold=A --fold_config=incountry --eval_folds=C

if __name__ == '__main__':
    args = _parse_args()
    infer_model(**vars(args))
    exit(0)


