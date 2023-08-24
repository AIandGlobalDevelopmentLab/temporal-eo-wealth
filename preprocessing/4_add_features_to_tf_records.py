from data_handling import create_full_tfrecords_paths
from glob import glob
import argparse
import time
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import dask.dataframe as dd
    
FRAME_PX_DIAMETER = 224
N_FRAMES_IN_TS_IMG = 10

def setup_new_directory(data_path, new_dir_name):
    dhs_tfrecord_dirs = glob(os.path.join(data_path, 'dhs_tfrecords', '*/'))
    new_dhs_tfrecord_dirs = [x.replace('dhs_tfrecords', new_dir_name) for x in dhs_tfrecord_dirs]

    new_dir_path = os.path.join(data_path, new_dir_name)
    os.mkdir(new_dir_path)
    for x in new_dhs_tfrecord_dirs:
        os.mkdir(x)
        

def parse_single_example(example_proto):
    """
    Parses a single tf-record file to readable dict with Tensor band values.
    :param example_proto: tf.train.Example, the tf-record example to parse
    :param bands: list, a list containg names for all the bands to use in image
    :param labeled: bool, should parsed example include "IWI" label?
    :param n_year_composites: int, number of image frames in tf-record
    :return: A dict mapping bands, year and possibly label to Tensors values
    """
    bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']
    keys_to_features = {}
    for band in bands:
        keys_to_features[band] = tf.io.FixedLenFeature(shape=[FRAME_PX_DIAMETER * FRAME_PX_DIAMETER * N_FRAMES_IN_TS_IMG], dtype=tf.float32)
    scalar_float_keys = ['year', 'iwi', 'lat', 'lon']
    for key in scalar_float_keys:
        keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
    return tf.io.parse_single_example(example_proto, features=keys_to_features)

def _floats_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_expanded_tfrecord(row):
    infile = row['file']
    
    # Parse sample from tf record
    raw_dataset = tf.data.TFRecordDataset([infile], compression_type='GZIP')
    x = next(iter(raw_dataset))
    x = parse_single_example(x)
    
    # Serialize sample
    bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']
    feature = {}
    for band in bands:
        value = x[band].numpy()
        feature[band] = _floats_feature(value=value)
    
    # Add values from df
    scalar_float_keys = ['year', 'iwi', 'lon', 'lat', 'rural', 'households']
    for key in scalar_float_keys:
        value = [tf.constant(row[key], dtype=tf.float32)]
        # Make sure the values in the df match what's in the tf-record
        if key in ['year', 'iwi', 'lon', 'lat']:
            if tf.math.abs(x[key] - value) > 0.0001:
                tf_scalar_values = [x[i].numpy() for i in ['year', 'iwi', 'lon', 'lat']]
                print('{}, {}, {}, {}, {}'.format(*([row.name] + tf_scalar_values)), flush=True)
                return ''
        feature[key] = _floats_feature(value=value)
    
    # Add df index
    feature['index'] = _int_feature(value=[tf.constant(row.name, dtype=tf.int64)])
        
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    example = example_proto.SerializeToString()
    
    outfile = infile.replace('dhs_tfrecords', NEW_DIR_NAME)
    # Write serialized sample to tf record
    with tf.io.TFRecordWriter(outfile, options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
        writer.write(example)
    return outfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='The path to the data directory')
    parser.add_argument(
        '--new-dir-name',
        type=str,
        default='expanded_tfrecords',
        help='Name of the directory which the results should be saved to.')
    parser.add_argument(
        '--num-splits',
        type=int,
        default=10,
        help='Number of shards to split the df into')
    parser.add_argument(
        '--split-id',
        type=int,
        required=True,
        help='The shard this worker should process')
    parser.add_argument(
        '--num-partitions',
        type=int,
        default=8,
        help='Number of threads this worker should use')
    args = parser.parse_args()
    
    DATA_PATH = args.data_path
    NEW_DIR_NAME = args.new_dir_name
    num_splits = args.num_splits
    split_id = args.split_id
    num_partitions = args.num_partitions

    new_dir_path = os.path.join(DATA_PATH, NEW_DIR_NAME)
    if not os.path.exists(new_dir_path):
        setup_new_directory(DATA_PATH, NEW_DIR_NAME)
    
    cluster_file = os.path.join(DATA_PATH, 'dhs_clusters.csv')

    df = pd.read_csv(cluster_file, float_precision='high', index_col=False)
    tfrecord_files = np.asarray(create_full_tfrecords_paths(DATA_PATH))
    df['file'] = tfrecord_files
    
    # Get this worker's shard
    idxs = list(range(split_id, len(df), num_splits))
    split_df = df.iloc[idxs]
    print(f'Starting to process {len(idxs)} records...')

    s_t = time.time()
    
    ddf = dd.from_pandas(split_df, npartitions=num_partitions)
    new_files = ddf.apply(create_expanded_tfrecord, axis=1, meta=('outfile', str)).compute()

    print(f'Finished in {time.time()-s_t} seconds')
    
    print('New files:', new_files)
    
    
    
