import argparse
import os
from pprint import pprint
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.stopper import MaximumIterationStopper
from ray.tune.integration.wandb import WandbLoggerCallback

from models.hp_tune_model import HPTuneModel
from models.resnet_lstm import ResNetLstm
from models.resnet_transformer import ResNetTransformer
from models.resnet import ResNet


def pbt_tune_model(**params: Any) -> None:
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

    # get all data params
    data_dir = params['data_dir']
    fold_config = params['fold_config']
    fold = params['fold']
    
    model_name = params['model_name']

    # General parameters
    config = {
        'name': model_name,
        'num_outputs': 1,
        'n_of_frames': params['n_of_frames'],
        'ms_channels': ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1'],
        'epochs': params['max_epochs'],
        'eval_frequency': 1,
        'batch_size': 64,
        'fold': fold,
        'fold_config': fold_config,
        'data_dir': data_dir
    }

    # Model type specific parameters
    if model_name == 'resnet':
        config['model_class'] = ResNet
        config['n_of_frames'] = 1
        hyperparam_mutations = {
            # 'l2': tune.loguniform(1e-4, 1),  # lambda: 10 ** np.random.randint(-4, 0)
            'lr': tune.loguniform(1e-10, 1e-1)  # lambda: 10 ** np.random.randint(-10, 0)
        }
    elif model_name == 'resnet_transformer':
        config['model_class'] = ResNetTransformer
        config['embed_dim'] = 400
        config['num_heads'] = 10
        config['num_layers'] = 3
        config['ff_dim'] = 250
        hyperparam_mutations = {
            # 'dropout': tune.uniform(0, 1),  # lambda: np.random.uniform(0, 1),
            'lr': tune.loguniform(1e-10, 1e-1)  # lambda: 10 ** np.random.randint(-10, 0)
        }
    elif model_name == 'bidirectional_resnet_lstm':
        config['model_class'] = ResNetLstm
        config['embed_dim'] = 400
        config['num_layers'] = 3
        config['ff_dim'] = 250
        config['n_lstm_units'] = 18
        config['bidirectional'] = True
        hyperparam_mutations = {
            # 'dropout': tune.uniform(0, 1),  # lambda: np.random.uniform(0, 1),
            'lr': tune.loguniform(1e-10, 1e-1),  # lambda: 10 ** np.random.randint(-10, 0)
            # 'momentum': tune.uniform(0.1, .9999)
        }
        if params['n_of_frames'] != 5:
            model_name = '{}_{}'.format(model_name, params['n_of_frames'])

    pbt = PopulationBasedTraining(
        time_attr='training_iteration',
        perturbation_interval=3,
        hyperparam_mutations=hyperparam_mutations)

    stopper = MaximumIterationStopper(max_iter=params['max_epochs'])

    n_cpu_cores = params['n_cpu_cores'] - 4 # Leave four CPU cores for overhead tasks
    n_gpus = params['n_gpus']
    n_cpus_per_task = n_cpu_cores // n_gpus
    
    name = '{}_{}_{}'.format(model_name, fold_config, fold)

    analysis = tune.run(
        HPTuneModel,
        name=name,
        time_budget_s=int(60 * 60 * 24 * params['time_budget_d']), # Convert to seconds
        scheduler=pbt,
        metric='val_loss',
        mode='min',
        stop=stopper,
        verbose=2,
        num_samples=n_gpus,
        resources_per_trial={
            'gpu': 1,
            'cpu': n_cpus_per_task
        },
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_score_attr='min-val_loss',
        max_failures=3,
        reuse_actors=True,
        log_to_file=True,
        local_dir=params['save_dir'],
        sync_config=tune.SyncConfig(syncer=None),
        config=config)

    '''
        callbacks=[WandbLoggerCallback(
            project='satellite_poverty_prediction_' + fold_config,
            group=name,
            api_key_file=os.path.join(data_dir, 'wandb_api_key_file'),
            log_config=True)],
        config=config)
    '''

    print_results(analysis)
    
    remove_mask = config['n_of_frames'] > 1
    best_model_path = save_best_model(analysis, name, params['save_dir'], remove_mask)
    print('The best model was saved at:', best_model_path, flush=True)


def print_results(analysis):
    stats = analysis.stats()
    wall_time = stats['timestamp'] - stats['start_time']
    hours = int(wall_time / 3600)
    minutes = int((wall_time - (hours * 3600)) / 60)
    val_loss = analysis.best_result['val_loss']
    val_r2 = analysis.best_result['val_r2']
    print(f'Experiment finished in {hours} hours and {minutes} minutes with a MSE of '  \
          f'{val_loss:.5f} and batch r^2 of {val_r2:.4f} on the validation set.')


def save_best_model(analysis, name, local_dir, remove_mask):
    # Load best model
    best_model = load_model(analysis.best_checkpoint.local_path, compile=False)
    
    # During training of multi-frame models, all frames without a label are masked out. 
    # During inference, we usually want outputs for all frames. This code removes that masking.
    if remove_mask:
        best_model = tf.keras.models.Model(inputs=best_model.inputs[0], 
                                           outputs=best_model.get_layer('unmasked_outputs').output)

    # Save best model in experiment_dir
    best_model_path = os.path.join(local_dir, name, 'best_model')
    save_model(best_model, best_model_path, include_optimizer=False)
    return best_model_path

def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run end-to-end population based training.')

    # paths
    parser.add_argument(
        '--data_dir', default='/mimer/NOBACKUP/groups/globalpoverty1/data',
        help='path to data directory')
    parser.add_argument(
        '--save_dir', default='/cephyr/NOBACKUP/groups/globalpoverty1/data/ray_results',
        help='path to save directory')
    parser.add_argument(
        '--fold', default='A', choices=['A', 'B', 'C', 'D', 'E'],
        help='fold to evaluate')

    # learning parameters
    parser.add_argument(
        '--label_name', default='iwi',
        help='name of label to use from the TFRecord files')
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Patience for early stopping')

    # data params
    parser.add_argument(
        '--train_frac', type=float, default=1.0,
        help='Fraction of training set to use')
    parser.add_argument(
        '--val_frac', type=float, default=1.0,
        help='Fraction of validation set to use')

    # high-level model control
    parser.add_argument(
        '--model_name', default='bidirectional_resnet_lstm', choices=['resnet_lstm', 'resnet', 'bidirectional_resnet_lstm', 'resnet_transformer'],
        help='name of model architecture')
    parser.add_argument(
        '--fold_config', default='incountry', choices=['incountry', 'ooc', 'oots'],
        help='What folds to use during training. Either \'incountry\', \'ooc\' or \'oots\'.')
    parser.add_argument(
        '--n_of_frames', type=int, default=10,
        help='Window size to use when training the model, i.e. nr of year composites')

    # Ray tune parameters
    parser.add_argument(
        '--tune_locally', action='store_true',
        help='Tune the model locally, i.e. not on a Ray tune cluster')
    parser.add_argument(
        '--n_gpus', type=int, default=12,
        help='number of GPUs in training cluster')
    parser.add_argument(
        '--n_cpu_cores', type=int, default=192,
        help='number of CPU cores in training cluster')
    parser.add_argument(
        '--time_budget_d', type=float, default=1.9,
        help='number of days for which to tune models')

    # Misc
    parser.add_argument(
        '--max_epochs', type=int, default=300,
        help='maximum number of epochs for training')
    parser.add_argument(
        '--seed', type=int, default=123,
        help='seed for random initialization and shuffling')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    print(vars(args))

    # Defaults to training on cluster. Can be trained on single machine for debugging
    if vars(args)['tune_locally']:
        ray.init()
    else:
        print(os.environ['ip_head'], os.environ['redis_password'])
        ray.init(address='auto', _node_ip_address=os.environ['ip_head'].split(':')[0], _redis_password=os.environ['redis_password'])
    
    if not ray.is_initialized():
        raise Exception('The Ray tune was not initialized correctly')

    pbt_tune_model(**vars(args))

    print('Shuting down ray tune...')
    ray.shutdown()
    print('Shutdown complete. Exiting program.')
    exit(0)
