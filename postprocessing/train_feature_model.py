import numpy as np
import os
from pprint import pprint
from typing import Any
import argparse

from sklearn.linear_model import RidgeCV
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model

N_FRAMES_IN_TS_IMG = 10

def train_feature_model(**params: Any) -> None:
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

    model_name = params['model_name']
    data_dir = params['data_dir']
    model_fold = params['model_fold']
    fold_config = params['fold_config']
    save_new_weights = params['save_new_weights']

    # Read model
    model_path = os.path.join(data_dir, 'ray_results', f'{model_name}_{fold_config}_{model_fold}', 'best_model')
    full_model = load_model(model_path)

    # Read feature data frame
    df_path = os.path.join(data_dir, 'features', fold_config, model_name, model_fold, 'features.csv')
    df = pd.read_csv(df_path).rename(columns={'Unnamed: 0': 'i'})

    # Get feature data
    X, y, w, CV, X_test, y_test, w_test, i_test = get_data(df, model_fold)

    # Train model
    ridge_model = RidgeCV(cv=CV).fit(X, y, w)

    # Get model with OG weights to compare
    og_ridge_model = get_og_model(full_model)

    # Evaluate model on test set
    test_score = ridge_model.score(X_test, y_test, w_test)
    og_test_score = og_ridge_model.score(X_test, y_test, w_test)

    # Print results
    print('Original test score:', og_test_score, flush=True)
    print('New test score:', test_score, flush=True)
    print(f'Improved performance by {100 * (test_score/og_test_score - 1):1f} %', flush=True)

    # Save tuned model
    if save_new_weights:
        new_model_path = df_path.replace('features.csv', 'tuned_model')
        save_tuned_model(full_model, ridge_model, new_model_path)

    # Save predictions
    predictions_path = df_path.replace('features.csv', 'predictions.csv')
    save_predictions(df, model_fold, ridge_model, X_test, predictions_path)


def get_data(df, model_fold, ooc=False):

    # Get columns with features
    feat_cols = [x for x in df.columns if x not in ['i', 'y', 'w', 'fold']]
    test_i = df['fold'] == model_fold

    # Split into test and train set
    X_test = df[feat_cols][test_i].values
    y_test = df['y'][test_i].values
    w_test = df['w'][test_i].values
    i_test = df['i'][test_i].values

    train_df = df[~test_i].reset_index(drop=True)
    X = train_df[feat_cols].values
    y = train_df['y'].values
    w = train_df['w'].values

    # Create cross validation sets
    CV = []
    train_folds = 'ABCDE'.replace(model_fold, '')
    for eval_fold in train_folds:
        eval_fold_i = np.where(train_df['fold'] == eval_fold)[0]
        train_fold_i = np.where(train_df['fold'] != eval_fold)[0]
        CV.append((train_fold_i, eval_fold_i))

    return X, y, w, CV, X_test, y_test, w_test, i_test

def get_og_model(full_model):
    rr_layer = 'ridge_regression' if full_model.name == 'resnet' else 'dense_logits'
    full_model.summary()
    print(full_model.summary())
    og_weights = full_model.get_layer(rr_layer).get_weights()

    # Create ridge regressor and set weights to be same as deep model
    og_ridge_model = RidgeCV()
    og_ridge_model.coef_ = og_weights[0].flatten()
    og_ridge_model.intercept_ = og_weights[1]
    return og_ridge_model

def save_tuned_model(full_model, ridge_model, new_model_path):

    # Get ridge regression layer
    rr_layer = 'ridge_regression' if full_model.name == 'resnet' else 'dense_logits'

    # Set new weights
    tuned_coefs = np.expand_dims(ridge_model.coef_, 1)
    tuned_intercept = np.expand_dims(ridge_model.intercept_, 0)
    new_weights = [tuned_coefs, tuned_intercept]
    full_model.get_layer(rr_layer).set_weights(new_weights)

    # Save new model
    save_model(full_model, new_model_path, include_optimizer=False)
    print('Saved model to', new_model_path)

def save_predictions(df, model_fold, ridge_model, X_test, predictions_path):
    df_test = df[df['fold'] == model_fold]
    df_test['y_i'] = ridge_model.predict(X_test)
    pred_df = df_test.groupby('i').mean()[['y', 'y_i']]
    pred_df['fold'] = model_fold
    pred_df.to_csv(predictions_path)
    print('Predictions to', predictions_path)

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
        '--model_name', required=True, choices=['resnet', 'bidirectional_resnet_lstm', 'bidirectional_resnet_lstm_10'],
        help='name of model')
    parser.add_argument(
        '--model_fold', default='A', choices=['A', 'B', 'C', 'D', 'E'],
        help='CV fold used for traing model')
    parser.add_argument(
        '--fold_config', default='incountry', choices=['incountry', 'ooc', 'oots'],
        help='What folds was used during training. Either \'incountry\', \'ooc\' or \'oots\'.')
    parser.add_argument(
        '--save_new_weights', action='store_true',
        help='Save the optimal weights for model')

    # Misc
    parser.add_argument(
        '--seed', type=int, default=123,
        help='seed for random initialization and shuffling')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    train_feature_model(**vars(args))
    exit(0)
