'''
This script trains our ResNet-LSTM model to estimate wealth for DHS locations over time.
Model checkpoints and TensorBoard training logs are saved to `out_dir`.
Usage:
    python train_model.py \
        --label_name iwi \
        --model_name resnet \
        --lr_decay 0.96 --batch_size 64 \
        --eval_every 1 --print_every 40 \
        --max_epochs {max_epochs} \
        --n_lstm_units 18 \ --seed {seed} \
        --out_dir {out_dir} \ --data_dir {data_dir} \
        --experiment_name {experiment_name} \
        --ls_bands {ls_bands} \
        --lr {lr} --fc_reg {reg} --conv_reg {reg} \
        --init_resnet_lstm_dir {init_resnet_lstm_dir} \
        --init_lstm_dir {init_lstm_dir} \
        --init_resnet_dir {init_resnet_dir} \
        --hs_weight_init {hs_weight_init } \
	    --train_frac 0.8 \
	    --val_frac  0.2 \
	    --n_year_composites 10
Prerequisites: download TFRecords, process them, and create incountry folds. See
    `preprocessing/1_process_tfrecords.ipynb` and
    `preprocessing/2_create_incountry_folds.ipynb`.
'''
import argparse
import json
import os
import pickle
from pprint import pprint
from glob import glob
import time
from typing import Any, Dict, List, Optional
import data_handling
from models.ts_resnet_transformer import ResNetTransformer
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import tensorflow as tf

from batchers import mean_and_std_constants
from models.resnet_lstm import ResNetLstm
from models.resnet_transformer import ResNetTransformer
from models.resnet import ResNet

ROOT_DIR = os.path.dirname(__file__)  # folder containing this file

def run_training(
        data_dir: str,
        model_name: str,
        model_params: Dict[str, Any],
        batch_size: int,
        ls_bands: Optional[str],
        label_name: str,
        learning_rate: float,
        lr_decay: float,
        max_epochs: int,
        print_every: int,
        eval_every: int,
        out_dir: str,
        init_resnet_lstm_dir: Optional[str],
        init_lstm_dir: Optional[str],
        init_resnet_dir: Optional[str],
        train_frac: float,
        val_frac: float,
        n_year_composites: int,
        normalize: bool,
        pretext_task: str,
        size_of_window: int,
        patience: int,
        hs_weight_init: Optional[str]) -> None:
    '''
    Args
    - sess: tf.Session
    - ooc: bool, whether to use out-of-country split
    - dataset: str
    - keep_frac: float
    - model_name: str, currently only 'resnet' is supported
    - model_params: dict
    - batch_size: int
    - ls_bands: one of [None, 'rgb', 'ms']
    - nl_band: one of [None, 'merge', 'split']
    - label_name: str, name of the label in the TFRecord file
    - augment: bool
    - learning_rate: float
    - lr_decay: float
    - max_epochs: int
    - print_every: int
    - eval_every: int
    - num_threads: int
    - n_year_composites: int
    - cache: list of str, names of dataset splits to cache in RAM
    - out_dir: str, path to output directory for saving checkpoints and TensorBoard logs, must already exist
    - init_ckpt_dir: str, path to checkpoint dir from which to load existing weights
        - set to None to use ImageNet or random initialization
    - imagenet_weights_path: str, path to pre-trained weights from ImageNet
        - set to None to use saved ckpt or random initialization
    - hs_weight_init: str, one of [None, 'random', 'same', 'samescaled']
    - exclude_final_layer: bool, or None

    '''
    # ====================
    #    ERROR CHECKING
    # ====================
    assert os.path.exists(out_dir)

    if model_name == 'resnet_lstm' or model_name == 'bidirectional_resnet_lstm':
        model_class = ResNetLstm
        one_year_model = False
    elif model_name == 'resnet_transformer':
        model_class = ResNetTransformer
        one_year_model = False
    elif model_name == 'resnet':
        model_class = ResNet
        one_year_model = True
    else:
        raise ValueError('Unknown model_name. Only "resnet_lstm" model currently supported.')

    # ====================
    #       DATASET
    # ====================

    # get all tfrecords
    tfrecord_files = np.asarray(data_handling.create_full_tfrecords_paths(data_dir))

    # get train, val, test fold
    with open(data_dir + '/sorted_dhs_incountry_folds.pkl', 'rb') as pickle_file:
        content = pickle.load(pickle_file)

    # get band stats
    with open(data_dir + '/band_stats.json') as band_stats_file:
        band_stats = json.load(band_stats_file)

    # TODO: loop over all instead
    # select which fold to use
    folds = ['A', 'B', 'C', 'D', 'E']

    for fold in folds:

        test_indices = content[fold]['test']  # Never used
        train_indices = content[fold]['train']
        val_indices = content[fold]['val']

        # Sample according to the fraction specified by train_frac and val_frac
        train_indices = np.random.choice(train_indices, round(len(train_indices) * train_frac))
        val_indices = np.random.choice(val_indices, round(len(val_indices) * val_frac))

        num_train = len(train_indices)
        num_val = len(val_indices)

        print('num_train:', num_train)
        print('num_val:', num_val)

        train_steps_per_epoch = int(np.ceil(num_train / batch_size))
        val_steps_per_epoch = int(np.ceil(num_val / batch_size))

        # test_files = tfrecord_files[test_indices] Is never used

        train_files = tfrecord_files[train_indices]
        val_files = tfrecord_files[val_indices]

        train_cache_file = os.path.join(data_dir, 'cached_datasets', model_name, f'{fold}_train_data')
        val_cache_file = os.path.join(data_dir, 'cached_datasets', model_name, f'{fold}_val_data')

        if pretext_task:
            ds = data_handling.get_self_supervised_dataset(train_files, batch_size)
            val_ds = data_handling.get_self_supervised_dataset(train_files, batch_size)
        else:
            ds = data_handling.get_train_dataset(train_files, batch_size, labeled=True, size_of_window=size_of_window,
                                     one_year_model=one_year_model, n_year_composites=n_year_composites,
                                     normalize=normalize, band_stats=band_stats[fold], cache_file=train_cache_file)
            val_ds = data_handling.get_train_dataset(val_files, batch_size, labeled=True, size_of_window=size_of_window,
                                         one_year_model=one_year_model, n_year_composites=n_year_composites,
                                         normalize=normalize, band_stats=band_stats[fold], cache_file=val_cache_file)


        # Set up a unique tensorboard directory for each job
        # Create a tensorboard callback for the model to provide logs to
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{out_dir}/tensorlog-log-{fold}", histogram_freq=1)

        # ====================
        #        MODEL
        # ====================
        print('Building model...', flush=True)
        model_params['num_outputs'] = 1
        # Use mixed precision in Keras (https://www.tensorflow.org/guide/mixed_precision)
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

        if ls_bands == 'rgb':
            ms_channels = ['BLUE', 'GREEN', 'RED']
        else:
            ms_channels = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']

        # TODO: Generalize in the other method?
        model_params['path_to_resnet_lstm_model'] = None
        model_params['size_of_window'] = size_of_window
        model_params['ms_channels'] = ms_channels
        model_params['path_to_ms_model'] = None
        model_params['path_to_nl_model'] = None
        model_params['pretrained_ms'] = True
        model_params['pretrained_nl'] = True
        model_params['trainable_resnet_block'] = True
        model_params['path_to_resnet_model'] = None
        model_params['path_to_resnet_transformer_model'] = None
        model_params['embed_dim'] = 1024
        model_params['num_heads'] = 2
        model_params['num_layers'] = 2
        model_params['ff_dim'] = 512

        # Distribute training across multiple GPUs (see https://www.tensorflow.org/guide/distributed_training)
        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():
            model = model_class.get_model_with_head(model_params)
        optimizer = tf.keras.optimizers.Adam(learning_rate=4.4527409928817476e-05)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error')

        checkpoint_path = out_dir + "/" + fold #+ '/model.hdf5'
        # Set up model saving
        ckpt_callback = ModelCheckpoint(filepath=checkpoint_path,
                                        save_best_only=True,
                                        verbose=1,
                                        monitor='val_loss',
                                        mode='min')
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

        print('Model built. Initiating training...', flush=True)
        print(model.summary())

        history = model.fit(ds,
                            validation_data=val_ds,
                            validation_freq=eval_every,
                            epochs=max_epochs,
                            steps_per_epoch=train_steps_per_epoch,
                            validation_steps=val_steps_per_epoch,
                            callbacks=[tensorboard_callback, ckpt_callback, es_callback])

        print('Finished training', flush=True)

        checkpoint_path = out_dir + "/" + fold + '/train_history_dict'
        with open(checkpoint_path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        # Delete cache files
        #cache_files = glob(train_cache_file + '*') + glob(val_cache_file + '*')
        #for file in cache_files:
        #    os.remove(file)


def run_training_wrapper(**params: Any) -> None:
    '''
    params is a dict with keys matching the arguments from _parse_args()
    '''
    start = time.time()
    print('Current time:', start)

    # print all of the flags
    pprint(params)

    # parameters that might be 'None'
    none_params = ['ls_bands', 'hs_weight_init', 'init_resnet_lstm_dir',
                   'init_lstm_dir', 'init_resnet_dir']
    for p in none_params:
        if params[p] == 'None':
            params[p] = None

    # set the random seeds
    seed = params['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # create the output directory if needed
    # TODO: Create more descriptive name
    full_experiment_name = params['experiment_name']
    out_dir = os.path.join(params['out_dir'], full_experiment_name)
    params_filepath = os.path.join(out_dir, 'params.json')
    if os.path.exists(params_filepath):
        print(f'Stopping. Found previous run at: {params_filepath}')
        return

    print(f'Outputs directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    with open(params_filepath, 'w') as config_file:
        json.dump(params, config_file, indent=4)

    model_params = {}

    if params['model_name'] == 'resnet_lstm':
        model_params['n_lstm_units'] = params['n_lstm_units']
        model_params['name'] = 'resnet_lstm'
        model_params['bidirectional'] = False
    elif params['model_name'] == 'bidirectional_resnet_lstm':
        model_params['n_lstm_units'] = params['n_lstm_units']
        model_params['name'] = 'bidirectional_resnet_lstm'
        model_params['bidirectional'] = True
    elif params['model_name'] == 'resnet_transformer':
        model_params['embed_dim'] = params['embed_dim']
        model_params['num_heads'] = params['num_heads']
        model_params['ff_dims'] = params['ff_dims']
        model_params['name'] = 'resnet_transformer'
    else:
        model_params['name'] = 'resnet'

    model_params['freeze_resnet'] = False
    model_params['l2'] = 0

    run_training(
        data_dir=params['data_dir'],
        model_name=params['model_name'],
        model_params=model_params,
        batch_size=params['batch_size'],
        ls_bands=params['ls_bands'],
        label_name=params['label_name'],
        learning_rate=params['lr'],
        lr_decay=params['lr_decay'],
        max_epochs=params['max_epochs'],
        print_every=params['print_every'],
        eval_every=params['eval_every'],
        out_dir=out_dir,
        init_resnet_lstm_dir=params['init_resnet_lstm_dir'],
        init_lstm_dir=params['init_lstm_dir'],
        init_resnet_dir=params['init_resnet_dir'],
        hs_weight_init=params['hs_weight_init'],
        train_frac=params['train_frac'],
        val_frac=params['val_frac'],
        n_year_composites=params['n_year_composites'],
        normalize=params['normalize'],
        pretext_task=params['pretext_task'],
        patience=params['patience'],
        size_of_window=params['size_of_window']
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
        '--experiment_name', default='new_experiment',
        help='name of experiment being run')
    parser.add_argument(
        '--out_dir', default=os.path.join(ROOT_DIR, 'outputs/'),
        help='path to output directory for saving checkpoints and TensorBoard '
             'logs')
    parser.add_argument(
        '--data_dir', default=os.path.join(ROOT_DIR, 'data/'),
        help='path to data directory')

    # initialization
    parser.add_argument(
        '--init_resnet_lstm_dir', default=None,
        help='optional path to model file to initialize ResNet-LSTM model')
    parser.add_argument(
        '--init_lstm_dir', default=None,
        help='optional path to model file to initialize LSTM block')
    parser.add_argument(
        '--init_resnet_dir', default=None,
        help='optional path to model file to initialize ResNet block')
    parser.add_argument(
        '--hs_weight_init', choices=[None, 'random', 'image_net_same_scale', 'image_net_keep_rgb'],
        help='method for initializing weights')
    parser.add_argument('--path_to_resnet_model', type=str, default=None, help='Path to pretrained resnet model')

    # learning parameters
    parser.add_argument(
        '--label_name', default='iwi',
        help='name of label to use from the TFRecord files')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='batch size')
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate for optimizer')
    parser.add_argument(
        '--lr_decay', type=float, default=1.0,
        help='Decay rate of the learning rate')
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Patience for early stopping')

    # high-level model control
    parser.add_argument(
        '--model_name', default='resnet_lstm', choices=['resnet_lstm', 'resnet', 'bidirectional_resnet_lstm', 'resnet_transformer'],
        help='name of model architecture')
    parser.add_argument(
        '--pretext_task', default=None, choices=[None, 'shuffle', 'in-painting'],
        help='name of pretext task to use')
    parser.add_argument(
        '--size_of_window', type=int, default=10,
        help='Window size to use when training the model, i.e. nr of year composites')

    # resnet_lstm only params
    parser.add_argument(
        '--n_lstm_units', type=int, default=18, choices=[18, 34, 50],
        help='number of LSTM units')

    # resnet_transformer only params
    parser.add_argument(
        '--embed_dim', type=int, default=512,
        help='size of vector embeding corresponding to a single image frame')
    parser.add_argument(
        '--num_heads', type=int, default=2,
        help='number of attention heads in transformer')
    parser.add_argument(
        '--ff_dims', type=int, default=32,
        help='number of nodes in transformer dense layer')

    # data params
    parser.add_argument(
        '--ls_bands', choices=[None, 'rgb', 'ms'],
        help='Landsat bands to use')
    parser.add_argument(
        '--n_year_composites', type=int, default=10,
        help='The number of year composites to use')
    # TODO: should this be asserted to be less than 10?
    parser.add_argument(
        '--train_frac', type=float, default=1.0,
        help='Fraction of training set to use')
    parser.add_argument(
        '--val_frac', type=float, default=1.0,
        help='Fraction of validation set to use')
    parser.add_argument(
        '--normalize', action='store_true',
        help='Normalize the satellite data')

    # Misc
    parser.add_argument(
        '--max_epochs', type=int, default=150,
        help='maximum number of epochs for training')
    parser.add_argument(
        '--eval_every', type=int, default=1,
        help='evaluate the model on the validation set after every so many '
             'epochs of training')
    parser.add_argument(
        '--print_every', type=int, default=40,
        help='print training statistics after every so many steps')
    parser.add_argument(
        '--seed', type=int, default=123,
        help='seed for random initialization and shuffling')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_training_wrapper(**vars(args))
