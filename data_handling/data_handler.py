import os
from glob import glob
import pandas as pd
import tensorflow as tf
import random
from typing import List, Dict

MIN_IWI = tf.constant([0], dtype=tf.float32)
MAX_IWI = tf.constant([100], dtype=tf.float32)
FRAME_PX_DIAMETER = 224  # Each frame has a 224 x 224 px resolution
N_FRAMES_IN_TS_IMG = 10  # A time-series image has ten frames, each corresponding to a three year period
START_YEAR = 1990  # The first frame represents 1990-1992
END_YEAR = 2019  # The last frame represents 2017-2019
SPAN_LENGTH = 3  # Each frame represents a span of three years


def create_full_tfrecords_paths(data_path):
    """
    Creates sorted list with paths to all tf-record files in dhs_clusters.csv
    :param data_path: str, path to directory containing data files
    :return: A list with paths to tf-record files
    """
    tfrecord_paths = []
    df = pd.read_csv(data_path + '/dhs_clusters.csv', float_precision='high', index_col=False)
    surveys = list(df.groupby(['country', 'survey_start_year']).groups.keys())
    for (country, year) in surveys:
        glob_path = os.path.join(data_path + '/dhs_tfrecords/', country + '_'
                                 + str(year), '*.tfrecord.gz')
        survey_files = glob(glob_path)
        tfrecord_paths.extend(sorted(survey_files))
    return tfrecord_paths


def get_train_dataset(tfrecord_files: List[str], batch_size: int, n_of_frames: int = 10,
                      shuffle: bool = True, normalize: bool = True, band_stats: Dict = None, 
                      max_epochs: int = 300, augment = False):
    """
    Creates a dataset with for the model. Can be used for either training or inference, depending on the value of labeled.
    :param tfrecord_files: List[str], list of paths to all tf-records to include in the dataset
    :param batch_size: int, batch size for dataset
    :param labeled: bool, wheter the dataset should include "IWI" label or not
    :param n_of_frames: int, the number of frames to include in the window which the model gets to see. 
        For a more thourough explenation, check out TODO: Add markup file describing windows
    :param bands: List[str], name of all bands to include in images
    :param cache_file: str, path to file where cached results will be stored. Can speed up read times
    :param shuffle: bool, wheter to shuffle the dataset between epochs
    :param normalize: bool, wheter to normalize the band values or not
    :param band_stats: dict, constants (mean and var) used to normalize band values
    :param max_epochs: int, max number of times the dataset will be read
    :return: tf.Dataset, a dataset containing processed information from all the tf-records in tfrecord_files.
    """
    if normalize:
        assert band_stats != None, 'Can not normalize without band_stats'

    # Defaults to all bands
    bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']

    # Shuffle files to avoid batches containing only samples from a single country
    if shuffle:
        random.shuffle(tfrecord_files)

    # convert to individual records
    dataset = tf.data.TFRecordDataset(
        filenames=tfrecord_files,
        compression_type = 'GZIP',
        buffer_size = 1024 * 1024 * 128,  # 128 MB buffer size
        num_parallel_reads = 4)

    # prefetch 2 batches at a time to smooth out the time taken to
    # load input files as we go through shuffling and processing
    dataset = dataset.prefetch(buffer_size = 2 * batch_size)
    
    # Process all tf-record files into a readable format
    dataset = dataset.map(lambda x: get_labeled_sample(x, n_of_frames = n_of_frames, bands = bands, 
                            normalize = normalize, band_stats = band_stats, augment=augment),
                        num_parallel_calls=4)

    # batch then repeat => batches respect epoch boundaries
    if shuffle:
        dataset = dataset.shuffle(200)
    if batch_size > 0:
        dataset = dataset.batch(batch_size)
    if max_epochs > 1:
        dataset = dataset.repeat(max_epochs)

    # prefetch 2 batches at a time
    dataset = dataset.prefetch(buffer_size = 2)

    return dataset

@tf.function
def get_labeled_sample(example_proto, n_of_frames, bands, normalize=True, band_stats=None, augment=False):
    """
    Processes a tf-record file representing a DHS cluster such that it can be used as input for the models.
    :param example_proto: tf.train.Example, the tf-record example to parse
    :param labeled: bool, wheter processed example should include "IWI" label or not
    :param bands: List[str], name of all bands to include in images
    :param n_of_frames: int, the number of frames to include in the window which the model gets to see. 
        For a more thourough explenation, check out TODO: Add markup file describing windows
    :param normalize: bool, wheter to normalize the band values or not
    :param band_stats: dict, constants (mean and var) used to normalize band values
    :return: During training (when labeled == True), the function returns a dict containing:
            - model_input: A Tensor time-series image with the shape (n_of_frames, FRAME_PX_DIAMETER, FRAME_PX_DIAMETER, len(bands))
            - outputs_mask: A one-hot encoded tensor indicating the frame index where the survey took place
            - masked_outputs: Same as outputs_mask, but with the IWI value instead of 1. Used as label during training.
        During inference (when labeled == False), the function returns a dict containing:
            - model_input: A Tensor time-series image for each possible window configuration. 
                When 0 < n_of_frames < 10 this will result in multiple configurations (see markup file).
                The shape of model_input will be (n_of_congifurations, n_of_frames, FRAME_PX_DIAMETER, FRAME_PX_DIAMETER, len(bands))
            - outputs_mask: A one-hot encoded tensor indicating the frame index where the survey took place for each configuration
    """
    ex = parse_single_example(example_proto, bands)
    img = single_example_to_image(ex, bands, normalize, band_stats)
    if augment:
        img = augment_img(img)
    
    year = tf.cast(ex.get('year', -1), tf.int32)
    frame_index = get_frame_index_for_year(year)
    possible_start_indices = get_possible_start_indeces(frame_index, n_of_frames)
    
    iwi = tf.cast(ex.get('iwi', -1), tf.float32)
    normalized_iwi = (iwi - MIN_IWI) / (MAX_IWI - MIN_IWI)

    # During training, sample just one of the possible window configurations
    start_index = tf.random.shuffle(possible_start_indices)[0]
    one_hot_year = tf.one_hot((frame_index - start_index), n_of_frames, dtype=tf.float32)
    one_hot_label = normalized_iwi * one_hot_year
    
    img_splits = tf.split(img, [start_index, n_of_frames, 10 - (start_index + n_of_frames)], axis=0)
    window_img = img_splits[1]
    
    if n_of_frames > 1:
        sample_weight = tf.math.divide(n_of_frames, tf.math.reduce_sum(one_hot_year))
        return {'model_input': window_img, 'outputs_mask': one_hot_year}, {'masked_outputs': one_hot_label}, sample_weight
    else:
        # When only a single image frame is to be returned, the mask and time-axis are unnecessary.
        # They are therefore reomved before return.
        return tf.squeeze(window_img, axis=0), tf.squeeze(one_hot_label)

def get_inference_dataset(tfrecord_files: List[str], batch_size: int,
                      normalize: bool = True, band_stats: Dict = None, labeled = False):
    """
    Creates a dataset with for the model. Can be used for either training or inference, depending on the value of labeled.
    :param tfrecord_files: List[str], list of paths to all tf-records to include in the dataset
    :param batch_size: int, batch size for dataset
    :param labeled: bool, wheter the dataset should include "IWI" label or not
    :param n_of_frames: int, the number of frames to include in the window which the model gets to see. 
        For a more thourough explenation, check out TODO: Add markup file describing windows
    :param bands: List[str], name of all bands to include in images
    :param cache_file: str, path to file where cached results will be stored. Can speed up read times
    :param shuffle: bool, wheter to shuffle the dataset between epochs
    :param normalize: bool, wheter to normalize the band values or not
    :param band_stats: dict, constants (mean and var) used to normalize band values
    :param max_epochs: int, max number of times the dataset will be read
    :return: tf.Dataset, a dataset containing processed information from all the tf-records in tfrecord_files.
    """
    if normalize:
        assert band_stats != None, 'Can not normalize without band_stats'

    # Defaults to all bands
    bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']

    # convert to individual records
    dataset = tf.data.TFRecordDataset(
        filenames=tfrecord_files,
        compression_type = 'GZIP',
        buffer_size = 1024 * 1024 * 128,  # 128 MB buffer size
        num_parallel_reads = 4)

    # prefetch 2 batches at a time to smooth out the time taken to
    # load input files as we go through shuffling and processing
    dataset = dataset.prefetch(buffer_size = 2 * batch_size)
    
    # Process all tf-record files into a readable format
    dataset = dataset.map(lambda x: get_inference_sample(x, bands = bands, 
                            normalize = normalize, band_stats = band_stats, labeled=labeled),
                        num_parallel_calls=4)

    # batch then repeat => batches respect epoch boundaries
    # - i.e. last batch of each epoch might be smaller than batch_size
    if batch_size > 0:
        dataset = dataset.batch(batch_size)

    # prefetch 2 batches at a time
    dataset = dataset.prefetch(buffer_size = 2)

    return dataset

@tf.function
def get_inference_sample(example_proto, bands, normalize=True, band_stats=None, labeled=False):
    """
    Processes a tf-record file representing a DHS cluster such that it can be used as input for the models.
    :param example_proto: tf.train.Example, the tf-record example to parse
    :param labeled: bool, wheter processed example should include "IWI" label or not
    :param bands: List[str], name of all bands to include in images
    :param n_of_frames: int, the number of frames to include in the window which the model gets to see. 
        For a more thourough explenation, check out TODO: Add markup file describing windows
    :param normalize: bool, wheter to normalize the band values or not
    :param band_stats: dict, constants (mean and var) used to normalize band values
    :return: During training (when labeled == True), the function returns a dict containing:
            - model_input: A Tensor time-series image with the shape (n_of_frames, FRAME_PX_DIAMETER, FRAME_PX_DIAMETER, len(bands))
            - outputs_mask: A one-hot encoded tensor indicating the frame index where the survey took place
            - masked_outputs: Same as outputs_mask, but with the IWI value instead of 1. Used as label during training.
        During inference (when labeled == False), the function returns a dict containing:
            - model_input: A Tensor time-series image for each possible window configuration. 
                When 0 < n_of_frames < 10 this will result in multiple configurations (see markup file).
                The shape of model_input will be (n_of_congifurations, n_of_frames, FRAME_PX_DIAMETER, FRAME_PX_DIAMETER, len(bands))
            - outputs_mask: A one-hot encoded tensor indicating the frame index where the survey took place for each configuration
    """
    ex = parse_single_example(example_proto, bands)
    img = single_example_to_image(ex, bands, normalize, band_stats)
    
    year = tf.cast(ex.get('year', -1), tf.int32)
    frame_index = get_frame_index_for_year(year)

    sample_index = tf.cast(ex.get('index', -1), tf.int32)
    return_dict = {'sample_index': sample_index, 
                   'frame_index': frame_index, 
                   'model_input': img}
    
    if labeled:
        iwi = tf.cast(ex.get('iwi', -1), tf.float32)
        normalized_iwi = (iwi - MIN_IWI) / (MAX_IWI - MIN_IWI)
        return return_dict, normalized_iwi
    else:
        return return_dict

@tf.function
def parse_single_example(example_proto, bands):
    """
    Parses a single tf-record file to readable dict with Tensor band values.
    :param example_proto: tf.train.Example, the tf-record example to parse
    :param bands: list, a list containg names for all the bands to use in image
    :param labeled: bool, should parsed example include "IWI" label?
    :param n_year_composites: int, number of image frames in tf-record
    :return: A dict mapping bands, year and possibly label to Tensors values
    """
    keys_to_features = {}
    for band in bands:
        keys_to_features[band] = tf.io.FixedLenFeature(shape=[FRAME_PX_DIAMETER * FRAME_PX_DIAMETER * N_FRAMES_IN_TS_IMG], dtype=tf.float32)
    scalar_float_keys = ['year', 'iwi']
    for key in scalar_float_keys:
        keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
    keys_to_features['index'] = tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
    return tf.io.parse_single_example(example_proto, features=keys_to_features)

@tf.function
def single_example_to_image(x, bands, normalize, band_stats):
    """
    Converts a dict of Tensor band values to a correctly formated Tensor time-series image
    :param x: dict, a dict mapping bands to Tensors values
    :param bands: list, a list containg names for all the bands to use in image
    :param normalize: bool, wheter to normalize the band values or not
    :param band_stats: dict, constants (mean and var) used to normalize band values
    :return: A Tensor representation of the time-series image with the 
        shape (N_FRAMES_IN_TS_IMG, FRAME_PX_DIAMETER, FRAME_PX_DIAMETER, len(bands))
    """
    assert len(bands) > 0, 'Can\'t parse image since no bands where submitted'
    
    img = float('nan')
    ex = {}
    for band in bands:
        ex[band] = tf.nn.relu(x[band])
        ex[band] = tf.reshape(ex[band], (N_FRAMES_IN_TS_IMG, FRAME_PX_DIAMETER, FRAME_PX_DIAMETER))
    if normalize:
        means = band_stats['means']
        std_devs = band_stats['stds']
        norm_mins = band_stats['norm_mins']
        norm_maxs = band_stats['norm_maxs']
        for band in bands[:-1]:
            norm_band = (ex[band] - means[band]) / std_devs[band]
            ex[band] = (norm_band - norm_mins[band]) / (norm_maxs[band] - norm_mins[band])
        nl_band = 'NIGHTLIGHTS'
        ex[nl_band] = tf.reshape(ex[nl_band], (N_FRAMES_IN_TS_IMG, FRAME_PX_DIAMETER, FRAME_PX_DIAMETER))
        dmsp, viirs = tf.split(ex[nl_band], [8, 2], axis=0)

        norm_dmsp = (dmsp - means['DMSP']) / std_devs['DMSP']
        norm_viirs = (viirs - means['VIIRS']) / std_devs['VIIRS']
        scaled_dmsp = (norm_dmsp - norm_mins['DMSP']) / (norm_maxs['DMSP'] - norm_mins['DMSP'])
        scaled_viirs = (norm_viirs - norm_mins['VIIRS']) / (norm_maxs['VIIRS'] - norm_mins['VIIRS'])

        ex[nl_band] = tf.concat([scaled_dmsp, scaled_viirs], axis=0)

    img = tf.stack([ex[band] for band in bands], axis=3)
    return img

@tf.function
def augment_img(img, max_brightness_delta=0.2, max_contrast_delta=0.25):
    # Apply random flips
    img = tf.cond(tf.random.uniform(shape=()) > 0.5, lambda: tf.image.flip_left_right(img), lambda: img)
    img = tf.cond(tf.random.uniform(shape=()) > 0.5, lambda: tf.image.random_flip_up_down(img), lambda: img)

    # Apply random brightness/contrast shift to non-nightlight bands
    ms_bands = img[:, :, :, :-1]
    ms_bands = tf.image.random_brightness(ms_bands, max_delta=max_brightness_delta)
    ms_bands = tf.image.random_contrast(ms_bands, lower=1-max_contrast_delta, upper=1+max_contrast_delta)

    return tf.concat([ms_bands, img[:, :, :, -1:]], axis=3)

@tf.function
def get_frame_index_for_year(year):
    """
    Finds the frame index (0-9) which corresponds to the given year.
    """
    # assert START_YEAR <= year and year <= END_YEAR, f'The year ({year} is not in the range ({START_YEAR}-{END_YEAR})'
    index = (year - START_YEAR) // SPAN_LENGTH
    return index

@tf.function
def get_possible_start_indeces(frame_index, n_of_frames):
    """
    Given a frame index (0-9) and a window size, the function returns the start 
    indices for all possible window configurations which includes the given index.
    See markup file for more information.
    """
    min_possible_i = tf.math.maximum(0, frame_index - n_of_frames + 1)
    max_possible_i = tf.math.minimum(10 - n_of_frames, frame_index)
    return tf.range(min_possible_i, max_possible_i + 1)
