import json
import os
import pickle
import data_handling
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from ray import tune

from data_handling import get_train_dataset
from utils.metrics import r2

# Hide unnecessary warnings
try:
    tf.get_logger().setLevel('WARNING')
except Exception as exc:
    print(exc)
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', message=r'WARNING:absl:Found untraced functions such as*')
warnings.filterwarnings('ignore', message=r'WARNING trial_runner.py:1408 -- You are trying to access _search_alg*')

class HPTuneModel(tune.Trainable):
    """
    Wrapper class for tuning model hyperparameters with PBT
    """

    def get_fold_data(self):
        # get all tfrecords
        data_dir = self.config['data_dir']
        tfrecord_files = np.asarray(data_handling.create_full_tfrecords_paths(data_dir))

        # get train, val, test fold
        fold_config = self.config['fold_config']
        fold = self.config['fold']
        folds_file_path = os.path.join(data_dir, 'folds', fold_config + '.pkl')
        with open(folds_file_path, 'rb') as pickle_file:
            content = pickle.load(pickle_file)

        # get band stats
        stats_file_path = os.path.join(data_dir, 'band_stats', fold_config + '.json')
        with open(stats_file_path) as band_stats_file:
            band_stats = json.load(band_stats_file)[fold]
        
        # select which fold to use
        train_indices = content[fold]['train']
        val_indices = content[fold]['val']

        # Sample according to the fraction specified by train_frac and val_frac
        train_frac = self.config.get('train_frac', 1.0)
        val_frac = self.config.get('val_frac', 1.0)
        train_indices = np.random.choice(train_indices, round(len(train_indices) * train_frac))
        val_indices = np.random.choice(val_indices, round(len(val_indices) * val_frac))

        num_train = len(train_indices)
        num_val = len(val_indices)

        print('num_train:', num_train)
        print('num_val:', num_val)

        train_files = tfrecord_files[train_indices]
        val_files = tfrecord_files[val_indices]

        return train_files, val_files, band_stats
    
    def setup_ds(self):
        train_files, val_files, band_stats = self.get_fold_data()
        batch_size = self.config['batch_size']

        ds = get_train_dataset(train_files, batch_size, n_of_frames=self.config['n_of_frames'], 
                                 normalize=True, band_stats=band_stats, max_epochs=self.config['epochs'], augment=True)

        val_ds = get_train_dataset(val_files, 8, n_of_frames=self.config['n_of_frames'], 
                                     normalize=True, band_stats=band_stats, max_epochs=self.config['epochs'])
                                     
        num_train = len(train_files)
        num_val = len(val_files)

        train_steps_per_epoch = int(np.ceil(num_train / batch_size))
        val_steps_per_epoch = int(np.ceil(num_val / batch_size))
        
        return ds, val_ds, train_steps_per_epoch, val_steps_per_epoch

    def setup_model(self):
        # Use mixed precision in Keras (https://www.tensorflow.org/guide/mixed_precision)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        model_class = self.config['model_class']

        model = model_class.get_model_with_head(self.config)
        
        optimizer = Adam(learning_rate=self.config['lr'])
        # optimizer = SGD(learning_rate=config['lr'], momentum=config['momentum'])
    
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=[r2])
        
        return model

    def setup(self, config):

        self.config = config
        self.model = self.setup_model()
        self.ds, self.val_ds, self.train_steps_per_epoch, self.val_steps_per_epoch = self.setup_ds()

    def step(self):

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.config['lr'])

        history = self.model.fit(self.ds, steps_per_epoch=self.train_steps_per_epoch, verbose=0)
        
        val_loss, val_r2 = self.model.evaluate(self.val_ds, steps=self.val_steps_per_epoch, verbose=0)
        
        res = {
            'loss': history.history['loss'][-1],
            'r2': history.history['r2'][-1],
            'val_loss': val_loss, 
            'val_r2': val_r2
            }

        '''
        nan_metrics = []
        for key in res:
            if tf.math.is_nan(res[key]):
                nan_metrics.append(key)
                
        if nan_metrics:
            raise ValueError(f'The following metrics returned with nan values: {nan_metrics}')
        '''

        print(res, flush=True)

        return res

    def save_checkpoint(self, checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, 'model')
        print('Save checkpoint:', file_path, flush=True)
        tf.keras.models.save_model(self.model, file_path) 
        return file_path

    def load_checkpoint(self, path):
        # See https://stackoverflow.com/a/42763323
        del(self.model)
        self.model = load_model(path, custom_objects={'r2': r2})
        self.model.compile(loss=self.model.loss, optimizer=self.model.optimizer, metrics=[r2])
        print('Load checkpoint:', path, flush=True)

    def reset_config(self, new_config):
        print('Reset config...')
        del(self.model)
        self.config = new_config
        self.model = self.setup_model()

        return True